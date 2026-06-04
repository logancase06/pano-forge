"""Tests pour l'étape d'ingestion (:mod:`src.ingest`)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.ingest import (
    MIN_IMAGES,
    IngestError,
    find_images,
    ingest,
    load_and_fix_orientation,
)


def _make_image(path, size=(64, 48), color=(120, 30, 200), orientation=None):
    """Écrit une image JPEG de test, avec un tag EXIF Orientation optionnel."""
    img = Image.new("RGB", size, color)
    exif = img.getexif()
    if orientation is not None:
        exif[0x0112] = orientation
        img.save(path, format="JPEG", exif=exif)
    else:
        img.save(path, format="JPEG")
    return path


def _populate(directory, count, **kwargs):
    for i in range(count):
        _make_image(directory / f"photo_{i:02d}.jpg", **kwargs)


def test_find_images_sorted_and_filtered(tmp_path):
    _make_image(tmp_path / "b.jpg")
    _make_image(tmp_path / "a.jpg")
    (tmp_path / "notes.txt").write_text("pas une image")
    (tmp_path / "sub").mkdir()  # ignoré : pas récursif

    found = find_images(tmp_path)

    assert [p.name for p in found] == ["a.jpg", "b.jpg"]


def test_find_images_missing_dir(tmp_path):
    with pytest.raises(IngestError):
        find_images(tmp_path / "n_existe_pas")


def test_ingest_requires_minimum_images(tmp_path):
    _populate(tmp_path, MIN_IMAGES - 1)
    with pytest.raises(IngestError, match="Pas assez d'images"):
        ingest(tmp_path)


def test_ingest_empty_dir(tmp_path):
    with pytest.raises(IngestError, match="Aucune image"):
        ingest(tmp_path)


def test_ingest_success(tmp_path):
    _populate(tmp_path, MIN_IMAGES)
    results = ingest(tmp_path)

    assert len(results) == MIN_IMAGES
    for item in results:
        assert item.width > 0 and item.height > 0
        assert item.image.mode == "RGB"


def test_ingest_writes_output(tmp_path):
    _populate(tmp_path, MIN_IMAGES)
    out = tmp_path / "out"
    ingest(tmp_path, output_dir=out)

    written = sorted(p.name for p in out.glob("*.jpg"))
    assert len(written) == MIN_IMAGES


def test_orientation_correction_swaps_dimensions(tmp_path):
    # Orientation 6 = rotation 90° : largeur et hauteur doivent être échangées.
    path = _make_image(tmp_path / "rot.jpg", size=(80, 40), orientation=6)
    image, rotated = load_and_fix_orientation(path)

    assert rotated is True
    assert (image.width, image.height) == (40, 80)


def test_no_orientation_tag_keeps_dimensions(tmp_path):
    path = _make_image(tmp_path / "plain.jpg", size=(80, 40))
    image, rotated = load_and_fix_orientation(path)

    assert rotated is False
    assert (image.width, image.height) == (80, 40)


def test_corrupt_file_skipped_but_minimum_enforced(tmp_path):
    _populate(tmp_path, MIN_IMAGES)
    # Un fichier .jpg invalide doit être ignoré sans planter la collecte.
    (tmp_path / "broken.jpg").write_bytes(b"ceci n'est pas une image")

    results = ingest(tmp_path)
    assert len(results) == MIN_IMAGES
