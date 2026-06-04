"""Étape 1 du pipeline : ingestion des photos.

Lit les photos depuis un dossier d'entrée, corrige leur orientation à partir
des métadonnées EXIF, et valide qu'il y a assez d'images pour construire un
panorama (au moins 3 par défaut).

Usage en ligne de commande :

    python -m src.ingest input/ --output output/ingested

ou via la fonction :

    from src.ingest import ingest
    images = ingest("input")
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# Extensions reconnues comme photos d'entrée.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Nombre minimal d'images requis pour pouvoir assembler un panorama.
MIN_IMAGES = 3


class IngestError(Exception):
    """Erreur levée quand l'ingestion ne peut pas se faire (pas assez d'images, etc.)."""


@dataclass
class IngestedImage:
    """Une photo lue, normalisée et prête pour l'étape suivante du pipeline."""

    path: Path
    image: Image.Image
    width: int
    height: int
    exif_rotated: bool

    def __repr__(self) -> str:  # pragma: no cover - confort de debug
        return (
            f"IngestedImage(path={self.path.name!r}, size={self.width}x{self.height}, "
            f"exif_rotated={self.exif_rotated})"
        )


def find_images(input_dir: str | Path) -> list[Path]:
    """Retourne la liste triée des fichiers image présents dans ``input_dir``.

    La recherche n'est pas récursive : seules les images directement dans le
    dossier sont prises en compte. Le tri par nom rend l'ordre déterministe,
    ce qui aide pour l'assemblage et pour les tests.
    """
    directory = Path(input_dir)
    if not directory.exists():
        raise IngestError(f"Le dossier d'entrée n'existe pas : {directory}")
    if not directory.is_dir():
        raise IngestError(f"Le chemin d'entrée n'est pas un dossier : {directory}")

    images = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def load_and_fix_orientation(path: str | Path) -> tuple[Image.Image, bool]:
    """Ouvre une image et applique la rotation EXIF si nécessaire.

    Beaucoup d'appareils photo et de smartphones enregistrent l'image dans son
    orientation capteur brute et stockent l'orientation réelle dans le tag EXIF
    ``Orientation``. ``ImageOps.exif_transpose`` applique cette transformation et
    retourne une image correctement orientée, sans le tag d'orientation.

    Retourne ``(image, exif_rotated)`` où ``exif_rotated`` indique si une
    correction a effectivement été appliquée.
    """
    path = Path(path)
    try:
        img = Image.open(path)
        img.load()
    except UnidentifiedImageError as exc:
        raise IngestError(f"Fichier image illisible ou corrompu : {path}") from exc
    except OSError as exc:
        raise IngestError(f"Impossible d'ouvrir l'image : {path} ({exc})") from exc

    # Lit le tag d'orientation (0x0112) pour savoir si une correction s'applique.
    orientation = None
    exif = img.getexif()
    if exif is not None:
        orientation = exif.get(0x0112)
    exif_rotated = orientation not in (None, 1)

    # exif_transpose applique la rotation/miroir et nettoie le tag.
    fixed = ImageOps.exif_transpose(img)

    # Convertit en RGB pour homogénéiser les étapes suivantes (le PNG peut être
    # en RGBA/P, certains TIFF en CMYK, etc.).
    if fixed.mode != "RGB":
        fixed = fixed.convert("RGB")

    return fixed, exif_rotated


def ingest(
    input_dir: str | Path,
    *,
    min_images: int = MIN_IMAGES,
    output_dir: str | Path | None = None,
) -> list[IngestedImage]:
    """Ingère toutes les photos d'un dossier.

    1. liste les images supportées dans ``input_dir`` ;
    2. valide qu'il y en a au moins ``min_images`` ;
    3. ouvre chacune et corrige son orientation EXIF.

    Si ``output_dir`` est fourni, les images corrigées y sont écrites en JPEG.

    Lève :class:`IngestError` si le dossier est vide ou s'il y a moins de
    ``min_images`` images valides.
    """
    paths = find_images(input_dir)
    if not paths:
        raise IngestError(
            f"Aucune image trouvée dans {input_dir!s}. "
            f"Extensions supportées : {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    results: list[IngestedImage] = []
    errors: list[str] = []
    for path in paths:
        try:
            image, rotated = load_and_fix_orientation(path)
        except IngestError as exc:
            errors.append(str(exc))
            continue
        results.append(
            IngestedImage(
                path=path,
                image=image,
                width=image.width,
                height=image.height,
                exif_rotated=rotated,
            )
        )

    if len(results) < min_images:
        detail = ""
        if errors:
            detail = " Erreurs rencontrées :\n  - " + "\n  - ".join(errors)
        raise IngestError(
            f"Pas assez d'images valides : {len(results)} trouvée(s), "
            f"{min_images} requise(s) au minimum.{detail}"
        )

    if output_dir is not None:
        _write_outputs(results, output_dir)

    return results


def _write_outputs(images: list[IngestedImage], output_dir: str | Path) -> None:
    """Écrit les images corrigées en JPEG dans ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for item in images:
        dest = out / f"{item.path.stem}.jpg"
        item.image.save(dest, format="JPEG", quality=95)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Lit les photos d'un dossier, corrige l'orientation EXIF "
        "et valide qu'il y a assez d'images pour un panorama.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="input",
        help="Dossier contenant les photos (par défaut : input/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        default=None,
        help="Dossier où écrire les images corrigées (optionnel).",
    )
    parser.add_argument(
        "-m",
        "--min-images",
        type=int,
        default=MIN_IMAGES,
        help=f"Nombre minimal d'images requis (par défaut : {MIN_IMAGES}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        images = ingest(
            args.input_dir,
            min_images=args.min_images,
            output_dir=args.output_dir,
        )
    except IngestError as exc:
        print(f"[ingest] Erreur : {exc}")
        return 1

    rotated = sum(1 for i in images if i.exif_rotated)
    print(f"[ingest] {len(images)} image(s) ingérée(s) depuis {args.input_dir!s}.")
    print(f"[ingest] {rotated} image(s) réorientée(s) via EXIF.")
    if args.output_dir:
        print(f"[ingest] Images corrigées écrites dans {args.output_dir!s}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
