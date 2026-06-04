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
import json
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

# Endpoint World Labs (placeholder tant que l'intégration n'est pas câblée).
WORLD_LABS_ENDPOINT = "https://api.worldlabs.ai/v1/worlds"


@dataclass
class PipelineResult:
    """Résultat complet d'un passage du pipeline."""

    prompt: str
    panorama: Panorama
    panorama_path: Path
    world_labs_request: dict = field(default_factory=dict)


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


def submit_world_labs(payload: dict):
    """Envoie la requête à World Labs pour générer le monde 3D navigable.

    Pas encore implémenté.
    """
    raise NotImplementedError(
        "L'envoi à World Labs n'est pas encore implémenté. "
        "Utilise prepare_world_labs() pour construire la requête."
    )


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
) -> PipelineResult:
    """Exécute le pipeline complet depuis un dossier de photos.

    Args:
        input_dir: dossier des photos sources.
        output_dir: dossier de sortie pour le panorama et le manifeste.
        backend: moteur panorama (``"hf_space"`` par défaut, ou ``"local_dit360"``).
        seed: graine de génération du panorama.
        num_inference_steps: nombre d'étapes de diffusion.
        min_images: nombre minimal de photos requis.
        client: client Anthropic injectable (sinon construit à la volée).
        extra_guidance: consignes de style supplémentaires pour la caption.

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

    return PipelineResult(
        prompt=prompt,
        panorama=panorama,
        panorama_path=panorama_path,
        world_labs_request=world_labs_request,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Pipeline pano-forge : photos d'intérieur → panorama 360° "
        "→ requête World Labs.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = forge(
            args.input_dir,
            args.output_dir,
            backend=args.backend,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
        )
    except (IngestError, CaptionError, PanoramaError) as exc:
        print(f"[forge] Erreur : {exc}")
        return 1

    print(f"[forge] Prompt : {result.prompt}")
    print(f"[forge] Panorama : {result.panorama_path}")
    print(
        f"[forge] Requête World Labs préparée "
        f"({args.output_dir}/world_labs_request.json)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
