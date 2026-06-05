"""Tests d'intégration du pipeline ``--mix`` de bout en bout, **sans mock**.

Une micro-vidéo synthétique est générée par OpenCV (frames bruitées : fort
mouvement + forte netteté), puis le pipeline complet est exécuté hors-ligne
(``submit=False``) : validation → extraction (cache) → assemblage MP4 → log.
Les services réseau (World Labs) ne sont pas sollicités.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from src import forge as forge_module
from src.forge import MIX_CODECS, forge


def _make_video(path, *, frames=30, fps=8.0, size=(640, 480), seed=0):
    """Écrit une vidéo .mp4 de bruit aléatoire (assez longue/nette pour valider)."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("encodeur mp4v indisponible")
    w, h = size
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    writer.release()
    return rng


@pytest.fixture
def mix_inputs(tmp_path):
    """Une vidéo valide + deux photos basse résolution (pour tester l'upscale)."""
    inp = tmp_path / "input"
    inp.mkdir()
    rng = _make_video(tmp_path / "clip.mp4", frames=30, fps=8.0, seed=7)
    for i in range(2):
        arr = rng.integers(0, 256, (240, 320, 3), dtype="uint8")
        Image.fromarray(arr).save(inp / f"photo_{i}.jpg")
    return inp, tmp_path / "clip.mp4", tmp_path / "out"


def test_mix_pipeline_end_to_end(mix_inputs):
    pytest.importorskip("cv2")
    inp, vid, out = mix_inputs

    result = forge(inp, out, mix=True, video=vid, frames=10, still_seconds=0.5)

    # Le MP4 combiné est bien produit.
    assert result.backend == "mix"
    mixed = result.source_images[0]
    assert mixed.name == "mixed.mp4"
    assert mixed.exists() and mixed.stat().st_size > 0
    assert result.world is None  # pas de --submit : aucun appel réseau

    # Le log structuré contient les métriques attendues.
    run_log = json.loads((result.run_dir / "run.log.json").read_text(encoding="utf-8"))
    assert run_log["backend"] == "mix"
    assert {"validate", "build_mix"} <= set(run_log["steps"])
    mix = run_log["mix"]
    assert mix["frames_extracted"] == 30
    assert 0 < mix["frames_retained"] <= 10
    assert mix["photos_total"] == 2
    assert mix["photos_kept"] == 2  # bruit => netteté élevée, aucune photo écartée
    assert mix["codec"] in MIX_CODECS
    assert mix["audio"] is False
    assert mix["mp4_bytes"] > 0
    assert mix["mean_sharpness"] > 0


def test_mix_pipeline_uses_frame_cache(mix_inputs):
    pytest.importorskip("cv2")
    inp, vid, out = mix_inputs

    forge(inp, out, mix=True, video=vid, frames=10, still_seconds=0.5)

    # Le cache de frames est créé sous output/.cache/<hash>/.
    metas = list((out / ".cache").glob("*/meta.json"))
    assert len(metas) == 1
    mtime_before = metas[0].stat().st_mtime

    # Une seconde soumission de la même vidéo réutilise le cache (meta non réécrit).
    second = forge(inp, out, mix=True, video=vid, frames=10, still_seconds=0.5)
    assert second.source_images[0].exists()
    assert metas[0].stat().st_mtime == mtime_before


def test_validate_video_rejects_bad_inputs(tmp_path):
    pytest.importorskip("cv2")
    from src.forge import VideoValidationError, validate_video

    # Mauvaise extension.
    bad_ext = tmp_path / "clip.avi"
    bad_ext.write_bytes(b"not a real video")
    with pytest.raises(VideoValidationError, match="Format"):
        validate_video(bad_ext)

    # Bonne extension mais illisible.
    unreadable = tmp_path / "broken.mp4"
    unreadable.write_bytes(b"not a real video")
    with pytest.raises(VideoValidationError):
        validate_video(unreadable)

    # Trop courte (1 frame) : durée hors bornes.
    short = tmp_path / "short.mp4"
    _make_video(short, frames=1, fps=8.0, seed=1)
    with pytest.raises(VideoValidationError, match="Durée|Résolution"):
        validate_video(short)


def test_validate_video_accepts_good_input(tmp_path):
    pytest.importorskip("cv2")
    from src.forge import validate_video

    good = tmp_path / "good.mp4"
    _make_video(good, frames=30, fps=8.0, seed=2)
    meta = validate_video(good)
    assert meta["width"] == 640 and meta["height"] == 480
    assert meta["duration"] >= forge_module.MIN_VIDEO_SECONDS
