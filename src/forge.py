"""Point d'entrée du pipeline pano-forge : photos → monde 3D World Labs.

Objectif : un monde 3D **fidèle à la pièce photographiée**. La fidélité vient du
conditionnement par image — World Labs Marble génère le monde directement à
partir d'une (ou plusieurs) image(s) réelle(s).

Deux backends :

- ``worldlabs`` (défaut) — envoie la/les **vraie(s) photo(s)** à World Labs en
  ``image_prompt`` (fidèle aux pixels). Avec ``--multi``, jusqu'à 4 meilleures
  photos en ``multi-image`` (upload media-asset, comme image-blaster).
- ``dit360`` — ancien chemin : caption (Qwen-VL) → panorama DiT360 (texte→pano,
  générique) → World Labs. Conservé en option.

Flux : ``ingest → [sélection photos | caption+DiT360] → World Labs → assets``.

CLI :

    python -m src.forge input/ --submit                 # photo réelle -> World Labs
    python -m src.forge input/ --multi --submit         # 4 photos -> multi-image
    python -m src.forge input/ --backend dit360 --submit  # ancien chemin DiT360
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from src.caption import CaptionError, describe_room
from src.ingest import MIN_IMAGES, IngestError, ingest
from src.panorama import (
    DEFAULT_SEED,
    DEFAULT_STEPS,
    Panorama,
    PanoramaError,
    generate_panorama,
)

# API World Labs (Marble). Auth via header WLT-Api-Key ; clé dans l'environnement.
WORLD_LABS_ENDPOINT = "https://api.worldlabs.ai/marble/v1"
WORLD_LABS_MODEL = "marble-1.1"
WORLD_LABS_API_KEY_ENV = "WORLD_LABS_API_KEY"

# marble-1.1 multi-image accepte au plus 4 images de prompt.
MULTI_IMAGE_LIMIT = 4

# Backends de pipeline.
BACKEND_WORLDLABS = "worldlabs"
BACKEND_DIT360 = "dit360"
DEFAULT_BACKEND = BACKEND_WORLDLABS

# Bord long max des photos envoyées (orientation EXIF déjà corrigée par ingest).
MAX_SOURCE_EDGE = 2048


class WorldLabsError(Exception):
    """Erreur levée lors d'un échec d'appel à l'API World Labs."""


@dataclass
class PipelineResult:
    """Résultat complet d'un passage du pipeline."""

    backend: str
    source_images: list[Path]
    prompt: str | None = None
    panorama: Panorama | None = None
    world_labs_request: dict = field(default_factory=dict)
    world: dict | None = None  # réponse World Labs si --submit
    world_assets: dict | None = None  # assets téléchargés si --submit


# ---------------------------------------------------------------------------
# Sélection / export des photos sources
# ---------------------------------------------------------------------------


def _color_histogram(pil_image: Image.Image, *, bins: int = 8) -> np.ndarray:
    """Histogramme couleur RGB normalisé — descripteur perceptuel léger.

    L'image est réduite (rapidité) ; on concatène les histogrammes des 3 canaux
    et on normalise. Deux vues très similaires (même mur, même cadrage) ont des
    histogrammes proches ; des angles différents divergent.
    """
    img = pil_image.convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float64)
    channels = [
        np.histogram(arr[:, :, c], bins=bins, range=(0, 255))[0] for c in range(3)
    ]
    hist = np.concatenate(channels)
    total = hist.sum()
    return hist / total if total else hist


def _diverse_indices(feats: list[np.ndarray], k: int) -> list[int]:
    """Farthest-point sampling : indices des ``k`` descripteurs les plus diversifiés.

    Distance L1 entre histogrammes. Amorçage déterministe sur le descripteur le
    plus éloigné de la moyenne (le plus distinctif), puis ajout glouton de
    l'image qui maximise la distance minimale à l'ensemble déjà choisi.
    """
    matrix = np.asarray(feats)
    n = len(matrix)

    def l1(i: int, j: int) -> float:
        return float(np.abs(matrix[i] - matrix[j]).sum())

    mean = matrix.mean(axis=0)
    seed = int(np.argmax(np.abs(matrix - mean).sum(axis=1)))
    selected = [seed]
    while len(selected) < k:
        best_idx, best_dist = None, -1.0
        for j in range(n):
            if j in selected:
                continue
            d = min(l1(j, s) for s in selected)
            if d > best_dist:
                best_dist, best_idx = d, j
        selected.append(best_idx)
    return selected


def _select_images(images: list, *, multi: bool) -> list:
    """Sélectionne les photos sources pour World Labs.

    - **single** : la plus haute résolution (meilleure qualité d'ancrage).
    - **multi** : les ``MULTI_IMAGE_LIMIT`` photos les plus **diversifiées** en
      vue/angle (histogrammes couleur + farthest-point sampling), pour couvrir
      au mieux la pièce plutôt que 4 quasi-doublons.
    """
    if not multi:
        return sorted(
            images, key=lambda im: (-(im.width * im.height), im.path.name.lower())
        )[:1]
    if len(images) <= MULTI_IMAGE_LIMIT:
        return list(images)
    feats = [_color_histogram(im.image) for im in images]
    return [images[i] for i in _diverse_indices(feats, MULTI_IMAGE_LIMIT)]


def _export_image(pil_image: Image.Image, dest: str | Path, *, max_edge: int = MAX_SOURCE_EDGE) -> Path:
    """Écrit une image (réduite si besoin) en JPEG vers ``dest``."""
    img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
    longest = max(img.width, img.height)
    if longest > max_edge:
        scale = max_edge / longest
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=92)
    return dest


def _even_azimuth(index: int, count: int) -> int:
    return round(index * 360 / count) if count else 0


# ---------------------------------------------------------------------------
# Appels HTTP World Labs
# ---------------------------------------------------------------------------


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


def _put_file(url: str, path: str | Path, *, method: str = "PUT", headers: dict | None = None) -> None:
    """Upload binaire d'un fichier vers une URL signée (media-asset)."""
    raw = Path(path).read_bytes()
    req = urllib.request.Request(url, data=raw, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise WorldLabsError(f"Upload media-asset échoué : {url} ({exc})") from exc


# ---------------------------------------------------------------------------
# Construction de la requête worlds:generate
# ---------------------------------------------------------------------------


def _image_prompt_base64(path: str | Path) -> dict:
    """Contenu image inline (data_base64) pour une requête single-image."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WorldLabsError(f"Image introuvable : {path} ({exc})") from exc
    extension = path.suffix.lstrip(".").lower() or "jpg"
    mime = "image/jpeg" if extension in ("jpg", "jpeg") else f"image/{extension}"
    return {
        "source": "data_base64",
        "data_base64": base64.standard_b64encode(raw).decode(),
        "extension": extension,
        "mime_type": mime,
    }


def _upload_media_asset(path: str | Path, *, api_key: str, transport, upload) -> str:
    """Prépare et upload un media-asset, retourne son id (flux image-blaster)."""
    path = Path(path)
    extension = path.suffix.lstrip(".").lower() or "png"
    prepare = transport(
        f"{WORLD_LABS_ENDPOINT}/media-assets:prepare_upload",
        api_key=api_key,
        method="POST",
        payload={"file_name": path.name, "kind": "image", "extension": extension},
    )
    asset = prepare.get("media_asset") or {}
    asset_id = (
        asset.get("media_asset_id")
        or asset.get("id")
        or prepare.get("media_asset_id")
    )
    upload_info = prepare.get("upload_info") or prepare.get("upload") or {}
    upload_url = upload_info.get("upload_url") or upload_info.get("url")
    upload_method = (upload_info.get("upload_method") or upload_info.get("method") or "PUT").upper()
    headers = upload_info.get("required_headers") or upload_info.get("headers") or {}
    if not asset_id or not upload_url:
        raise WorldLabsError(
            "Réponse prepare_upload sans media_asset_id ou upload_url."
        )
    upload(upload_url, path, method=upload_method, headers=headers)
    return asset_id


def _normalize_images(images) -> list[tuple[Path, int]]:
    """Normalise l'entrée en liste de ``(path, azimuth)``."""
    if isinstance(images, (str, Path)):
        return [(Path(images), 0)]
    items: list[tuple[Path, int]] = []
    seq = list(images)
    for i, entry in enumerate(seq):
        if isinstance(entry, (tuple, list)):
            path, azimuth = entry[0], int(entry[1])
        else:
            path, azimuth = entry, _even_azimuth(i, len(seq))
        items.append((Path(path), azimuth))
    return items


def _build_world_request(
    items: list[tuple[Path, int]],
    *,
    prompt: str | None,
    display_name: str,
    multi: bool,
    api_key: str,
    transport,
    upload,
) -> dict:
    """Construit le corps ``worlds:generate`` (single data_base64 ou multi-image)."""
    if multi or len(items) > 1:
        if len(items) > MULTI_IMAGE_LIMIT:
            raise WorldLabsError(
                f"Multi-image accepte au plus {MULTI_IMAGE_LIMIT} images "
                f"(reçu {len(items)})."
            )
        multi_prompt = []
        for path, azimuth in items:
            asset_id = _upload_media_asset(
                path, api_key=api_key, transport=transport, upload=upload
            )
            multi_prompt.append(
                {
                    "azimuth": azimuth,
                    "content": {"source": "media_asset", "media_asset_id": asset_id},
                }
            )
        world_prompt = {"type": "multi-image", "multi_image_prompt": multi_prompt}
    else:
        world_prompt = {"type": "image", "image_prompt": _image_prompt_base64(items[0][0])}

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
    images,
    *,
    prompt: str | None = None,
    display_name: str = "pano-forge",
    api_key: str | None = None,
    multi: bool = False,
    poll_interval: float = 15.0,
    max_polls: int = 240,
    _transport=None,
    _upload=None,
) -> dict:
    """Génère un monde 3D World Labs : submit puis polling jusqu'à ``done``.

    ``images`` : un chemin (single-image), ou une liste de chemins / de
    ``(chemin, azimuth)`` (multi-image). Logique portée d'``image-blaster``
    (``generate-world.mjs``) : single en ``data_base64`` inline, multi via
    upload media-asset. La clé est lue depuis ``api_key`` ou
    ``WORLD_LABS_API_KEY``. Retourne ``operation.response`` (assets du monde).
    """
    api_key = api_key or os.environ.get(WORLD_LABS_API_KEY_ENV)
    if not api_key:
        raise WorldLabsError(
            f"Clé World Labs manquante : définis {WORLD_LABS_API_KEY_ENV}."
        )

    transport = _transport or _request_json
    upload = _upload or _put_file
    items = _normalize_images(images)

    request = _build_world_request(
        items,
        prompt=prompt,
        display_name=display_name,
        multi=multi,
        api_key=api_key,
        transport=transport,
        upload=upload,
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


# ---------------------------------------------------------------------------
# Téléchargement des assets
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: str | Path) -> Path:
    """Télécharge ``url`` vers ``dest`` (stdlib urllib)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, OSError) as exc:
        raise WorldLabsError(f"Téléchargement échoué : {url} ({exc})") from exc
    return dest


def _ext_from_url(url: str, fallback: str) -> str:
    """Extension de fichier déduite d'une URL (sinon ``fallback``)."""
    return Path(urlparse(url).path).suffix or fallback


def download_world_assets(
    world: dict, output_dir: str | Path, *, prefix: str = "world"
) -> dict:
    """Télécharge les assets d'un monde World Labs dans ``output_dir``.

    Mêmes assets que ``image-blaster`` (``downloadWorldAssets``) : mesh GLB,
    pano équirectangulaire, thumbnail, et splats ``.spz`` (toutes résolutions).
    Retourne un dict des chemins locaux ({"glb", "pano", "thumbnail", "spz": {...}}).
    """
    assets = world.get("assets", {}) or {}
    out = Path(output_dir)
    result: dict = {"spz": {}}

    glb_url = (assets.get("mesh") or {}).get("collider_mesh_url")
    if glb_url:
        result["glb"] = _download_file(glb_url, out / f"{prefix}.glb")

    pano_url = (assets.get("imagery") or {}).get("pano_url")
    if pano_url:
        ext = _ext_from_url(pano_url, ".png")
        result["pano"] = _download_file(pano_url, out / f"{prefix}-pano{ext}")

    thumb_url = assets.get("thumbnail_url")
    if thumb_url:
        ext = _ext_from_url(thumb_url, ".webp")
        result["thumbnail"] = _download_file(thumb_url, out / f"{prefix}-thumbnail{ext}")

    spz_urls = (assets.get("splats") or {}).get("spz_urls") or {}
    for key, url in spz_urls.items():
        if not url:
            continue
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(key))
        result["spz"][key] = _download_file(url, out / f"{prefix}-{safe}.spz")

    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def forge(
    input_dir: str | Path,
    output_dir: str | Path = "output",
    *,
    backend: str = DEFAULT_BACKEND,
    multi: bool = False,
    seed: int = DEFAULT_SEED,
    num_inference_steps: int = DEFAULT_STEPS,
    min_images: int = MIN_IMAGES,
    submit: bool = False,
    client=None,
    extra_guidance: str | None = None,
    hf_token: str | None = None,
) -> PipelineResult:
    """Exécute le pipeline depuis un dossier de photos.

    Args:
        input_dir: dossier des photos sources.
        output_dir: dossier de sortie.
        backend: ``"worldlabs"`` (défaut, photo réelle → World Labs) ou
            ``"dit360"`` (caption → panorama DiT360 → World Labs).
        multi: en backend worldlabs, envoie jusqu'à 4 photos en multi-image.
        seed / num_inference_steps: paramètres DiT360 (backend dit360).
        min_images: nombre minimal de photos requis.
        submit: si vrai, génère réellement le monde World Labs (paie) et
            télécharge les assets.
        client: client gradio injectable (caption, backend dit360).
        extra_guidance: consignes de style pour la caption (backend dit360).
        hf_token: token HF optionnel pour la caption (backend dit360).
    """
    if backend not in (BACKEND_WORLDLABS, BACKEND_DIT360):
        raise ValueError(
            f"Backend inconnu : {backend!r}. "
            f"Choisis {BACKEND_WORLDLABS!r} ou {BACKEND_DIT360!r}."
        )

    images = ingest(input_dir, min_images=min_images)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompt: str | None = None
    panorama: Panorama | None = None

    if backend == BACKEND_DIT360:
        source = _select_images(images, multi=False)[0]
        prompt = describe_room(
            source, client=client, extra_guidance=extra_guidance, hf_token=hf_token
        )
        panorama = generate_panorama(
            prompt, seed=seed, num_inference_steps=num_inference_steps
        )
        source_images = [panorama.save(out / "panorama.jpg")]
        items = [(source_images[0], 0)]
        use_multi = False
    else:  # worldlabs : photo(s) réelle(s)
        selected = _select_images(images, multi=multi)
        source_images = [
            _export_image(im.image, out / f"source-{i:02d}.jpg")
            for i, im in enumerate(selected)
        ]
        n = len(source_images)
        items = [(p, _even_azimuth(i, n)) for i, p in enumerate(source_images)]
        use_multi = multi and n > 1

    manifest = {
        "endpoint": WORLD_LABS_ENDPOINT,
        "model": WORLD_LABS_MODEL,
        "backend": backend,
        "multi": use_multi,
        "source_images": [str(p) for p in source_images],
        "prompt": prompt,
    }
    (out / "world_labs_request.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    world = None
    world_assets = None
    if submit:
        submit_arg = items if use_multi else items[0][0]
        world = submit_world_labs(
            submit_arg,
            prompt=prompt,
            display_name=out.name or "pano-forge",
            multi=use_multi,
        )
        world_assets = download_world_assets(world, out)

    return PipelineResult(
        backend=backend,
        source_images=source_images,
        prompt=prompt,
        panorama=panorama,
        world_labs_request=manifest,
        world=world,
        world_assets=world_assets,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Pipeline pano-forge : photos d'interieur -> monde 3D World Labs.",
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
        choices=[BACKEND_WORLDLABS, BACKEND_DIT360],
        default=DEFAULT_BACKEND,
        help=f"Backend (par défaut : {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help=f"Backend worldlabs : envoie jusqu'à {MULTI_IMAGE_LIMIT} photos en multi-image.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Graine DiT360 (par défaut : {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--steps",
        dest="num_inference_steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Étapes de diffusion DiT360 (par défaut : {DEFAULT_STEPS}).",
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
        help=f"Génère le monde World Labs (nécessite {WORLD_LABS_API_KEY_ENV}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Le prompt/les chemins peuvent contenir des caractères hors cp1252 ; force
    # UTF-8 sur les consoles Windows pour éviter un UnicodeEncodeError.
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
            multi=args.multi,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            min_images=args.min_images,
            submit=args.submit,
        )
    except (IngestError, CaptionError, PanoramaError, WorldLabsError, ValueError) as exc:
        print(f"[forge] Erreur : {exc}")
        return 1

    print(f"[forge] Backend : {result.backend}")
    for path in result.source_images:
        print(f"[forge] Image source : {path}")
    if result.prompt:
        print(f"[forge] Prompt : {result.prompt}")

    if result.world is not None:
        assets = result.world.get("assets", {})
        print(f"[forge] Monde World Labs généré (assets : {list(assets)}).")
        downloaded = result.world_assets or {}
        for kind in ("glb", "pano", "thumbnail"):
            if downloaded.get(kind):
                print(f"[forge]   {kind}: {downloaded[kind]}")
        for key, path in (downloaded.get("spz") or {}).items():
            print(f"[forge]   spz[{key}]: {path}")
    else:
        print(
            f"[forge] Requête World Labs préparée "
            f"({args.output_dir}/world_labs_request.json). "
            f"Ajoute --submit (+ {WORLD_LABS_API_KEY_ENV}) pour générer le monde."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
