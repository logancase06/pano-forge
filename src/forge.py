"""Point d'entrée du pipeline pano-forge : photo → monde 3D World Labs.

Enchaîne les trois étapes et prépare l'envoi vers World Labs :

    ingest  →  caption  →  panorama  →  (sauvegarde output/)  →  World Labs (stub)

1. :mod:`src.ingest` lit et normalise les photos de ``input/`` ;
2. :mod:`src.caption` décrit la première photo en un prompt 360° ;
3. :mod:`src.panorama` génère l'équirectangulaire (DiT360) ;
4. le panorama est sauvegardé dans ``output/`` ;
5. :func:`prepare_world_labs` construit la requête World Labs (envoi non
   encore implémenté).

CLI :

    python -m src.forge input/ --output output --backend hf_space --seed 0
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from src.caption import CaptionError, describe_room
from src.ingest import MIN_IMAGES, IngestError, ingest
from src.panorama import (
    DEFAULT_SEED,
    DEFAULT_STEPS,
    HFSpaceBackend,
    Panorama,
    PanoramaError,
    generate_panorama,
)

# API World Labs (Marble). Auth via header WLT-Api-Key ; clé dans l'environnement.
WORLD_LABS_ENDPOINT = "https://api.worldlabs.ai/marble/v1"
WORLD_LABS_MODEL = "marble-1.1"
WORLD_LABS_API_KEY_ENV = "WORLD_LABS_API_KEY"


class WorldLabsError(Exception):
    """Erreur levée lors d'un échec d'appel à l'API World Labs."""


@dataclass
class PipelineResult:
    """Résultat complet d'un passage du pipeline."""

    prompt: str
    panorama: Panorama
    panorama_path: Path
    world_labs_request: dict = field(default_factory=dict)
    world: dict | None = None  # réponse World Labs si --submit


def prepare_world_labs(
    panorama: Panorama,
    panorama_path: str | Path,
    *,
    output_dir: str | Path,
) -> dict:
    """Construit la requête World Labs et écrit un manifeste JSON.

    Ne fait **pas** l'appel réseau (voir :func:`submit_world_labs`). Retourne le
    payload qui serait envoyé et l'écrit dans ``output_dir/world_labs_request.json``.
    """
    payload = {
        "endpoint": WORLD_LABS_ENDPOINT,
        "input_type": "equirectangular_panorama",
        "panorama_path": str(panorama_path),
        "width": panorama.width,
        "height": panorama.height,
        "prompt": panorama.prompt,
    }
    manifest = Path(output_dir) / "world_labs_request.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _request_json(url: str, *, api_key: str, method: str = "GET", payload: dict | None = None) -> dict:
    """Appel JSON minimal vers l'API World Labs (header WLT-Api-Key)."""
    data = None
    headers = {"WLT-Api-Key": api_key}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise WorldLabsError(
            f"World Labs {method} a échoué ({exc.code}) : {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise WorldLabsError(f"World Labs injoignable : {exc}") from exc
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError:
        return {}


def _build_world_request(
    panorama_path: str | Path,
    *,
    prompt: str | None = None,
    display_name: str = "pano-forge",
) -> dict:
    """Construit le corps ``worlds:generate`` à partir du panorama (image base64)."""
    path = Path(panorama_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WorldLabsError(f"Panorama introuvable : {path} ({exc})") from exc

    extension = path.suffix.lstrip(".").lower() or "jpg"
    mime = "image/jpeg" if extension in ("jpg", "jpeg") else f"image/{extension}"
    world_prompt = {
        "type": "image",
        "image_prompt": {
            "source": "data_base64",
            "data_base64": base64.standard_b64encode(raw).decode(),
            "extension": extension,
            "mime_type": mime,
        },
    }
    if prompt:
        world_prompt["text_prompt"] = prompt

    return {
        "display_name": display_name,
        "model": WORLD_LABS_MODEL,
        "world_prompt": world_prompt,
    }


def _operation_id(operation: dict) -> str:
    op_id = (
        operation.get("operation_id")
        or operation.get("id")
        or operation.get("name")
    )
    if not op_id:
        raise WorldLabsError("Réponse World Labs sans operation_id.")
    return str(op_id).split("/")[-1]


def submit_world_labs(
    panorama_path: str | Path,
    *,
    prompt: str | None = None,
    display_name: str = "pano-forge",
    api_key: str | None = None,
    poll_interval: float = 15.0,
    max_polls: int = 240,
    _transport=None,
) -> dict:
    """Génère un monde 3D World Labs depuis le panorama : submit puis polling.

    Logique portée de ``image-blaster`` (``generate-world.mjs``) : POST
    ``worlds:generate``, puis polling de ``operations/{id}`` jusqu'à ``done``.

    La clé est lue depuis ``api_key`` ou la variable d'environnement
    ``WORLD_LABS_API_KEY``. Retourne la réponse (``operation.response``)
    contenant les assets du monde (mesh, pano, splats…).
    """
    api_key = api_key or os.environ.get(WORLD_LABS_API_KEY_ENV)
    if not api_key:
        raise WorldLabsError(
            f"Clé World Labs manquante : définis {WORLD_LABS_API_KEY_ENV}."
        )

    transport = _transport or _request_json
    request = _build_world_request(
        panorama_path, prompt=prompt, display_name=display_name
    )

    operation = transport(
        f"{WORLD_LABS_ENDPOINT}/worlds:generate",
        api_key=api_key,
        method="POST",
        payload=request,
    )
    op_id = _operation_id(operation)

    polls = 0
    while not operation.get("done"):
        if polls >= max_polls:
            raise WorldLabsError(
                f"Délai dépassé en attendant World Labs (operation {op_id})."
            )
        time.sleep(poll_interval)
        operation = transport(
            f"{WORLD_LABS_ENDPOINT}/operations/{op_id}", api_key=api_key
        )
        polls += 1

    if operation.get("error"):
        raise WorldLabsError(
            f"Génération World Labs échouée : {operation['error']}"
        )
    response = operation.get("response")
    if not response:
        raise WorldLabsError("Opération World Labs terminée sans réponse.")
    return response


def forge(
    input_dir: str | Path,
    output_dir: str | Path = "output",
    *,
    backend: str = HFSpaceBackend.name,
    seed: int = DEFAULT_SEED,
    num_inference_steps: int = DEFAULT_STEPS,
    min_images: int = MIN_IMAGES,
    client=None,
    extra_guidance: str | None = None,
    submit: bool = False,
) -> PipelineResult:
    """Exécute le pipeline complet depuis un dossier de photos.

    Args:
        input_dir: dossier des photos sources.
        output_dir: dossier de sortie pour le panorama et le manifeste.
        backend: moteur panorama (``"hf_space"`` par défaut, ou ``"local_dit360"``).
        seed: graine de génération du panorama.
        num_inference_steps: nombre d'étapes de diffusion.
        min_images: nombre minimal de photos requis.
        client: client gradio injectable pour la caption (sinon construit).
        extra_guidance: consignes de style supplémentaires pour la caption.
        submit: si vrai, envoie réellement le panorama à World Labs
            (nécessite ``WORLD_LABS_API_KEY``) et attend le monde 3D.

    Returns:
        Un :class:`PipelineResult`.
    """
    images = ingest(input_dir, min_images=min_images)
    source = images[0]

    prompt = describe_room(source, client=client, extra_guidance=extra_guidance)

    panorama = generate_panorama(
        prompt,
        backend=backend,
        seed=seed,
        num_inference_steps=num_inference_steps,
    )

    out = Path(output_dir)
    panorama_path = panorama.save(out / "panorama.jpg")

    world_labs_request = prepare_world_labs(
        panorama, panorama_path, output_dir=out
    )

    world = None
    if submit:
        world = submit_world_labs(
            panorama_path,
            prompt=panorama.prompt,
            display_name=out.name or "pano-forge",
        )

    return PipelineResult(
        prompt=prompt,
        panorama=panorama,
        panorama_path=panorama_path,
        world_labs_request=world_labs_request,
        world=world,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Pipeline pano-forge : photos d'interieur -> panorama 360 "
        "-> requete World Labs.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="input",
        help="Dossier des photos sources (par défaut : input/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        default="output",
        help="Dossier de sortie (par défaut : output/).",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default=HFSpaceBackend.name,
        help=f"Moteur panorama (par défaut : {HFSpaceBackend.name}).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Graine de génération (par défaut : {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--steps",
        dest="num_inference_steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Étapes de diffusion (par défaut : {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "-m",
        "--min-images",
        type=int,
        default=MIN_IMAGES,
        help=f"Nombre minimal de photos requis (par défaut : {MIN_IMAGES}).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=f"Envoie le panorama à World Labs (nécessite {WORLD_LABS_API_KEY_ENV}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Le prompt généré peut contenir des caractères hors cp1252 ; force UTF-8
    # sur les consoles Windows pour éviter un UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    args = _build_parser().parse_args(argv)
    try:
        result = forge(
            args.input_dir,
            args.output_dir,
            backend=args.backend,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            min_images=args.min_images,
            submit=args.submit,
        )
    except (IngestError, CaptionError, PanoramaError, WorldLabsError) as exc:
        print(f"[forge] Erreur : {exc}")
        return 1

    print(f"[forge] Prompt : {result.prompt}")
    print(f"[forge] Panorama : {result.panorama_path}")
    if result.world is not None:
        assets = result.world.get("assets", {})
        pano_url = assets.get("imagery", {}).get("pano_url")
        print(f"[forge] Monde World Labs généré (assets : {list(assets)}).")
        if pano_url:
            print(f"[forge] Pano World Labs : {pano_url}")
    else:
        print(
            f"[forge] Requête World Labs préparée "
            f"({args.output_dir}/world_labs_request.json). "
            f"Ajoute --submit (+ {WORLD_LABS_API_KEY_ENV}) pour générer le monde."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
