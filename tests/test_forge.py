"""Tests pour le pipeline (:mod:`src.forge`).

Les étapes réseau (ingest/caption/panorama/World Labs) sont remplacées via
monkeypatch ou transport injecté : on teste l'orchestration et la logique
World Labs, pas les services externes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src import forge as forge_module
from src.forge import (
    MIX_CODECS,
    MULTI_IMAGE_LIMIT,
    WORLD_LABS_MODEL,
    PipelineResult,
    WorldLabsError,
    _assign_photos_to_keyframes,
    _auto_brightness_bgr,
    _build_video_request,
    _build_world_request,
    _image_quality_metrics,
    _motion_blur_bgr,
    _open_video_writer,
    _operation_id,
    _request_json,
    _rotate_bgr,
    _select_images,
    _select_motion_frames,
    _sharpness,
    _upscale_if_low_res,
    _world_status,
    download_world_assets,
    forge,
    submit_world_labs,
)
from src.panorama import Panorama


class FakeImg:
    """Doublure d'IngestedImage : porte une image PIL, des dimensions, un chemin."""

    def __init__(self, w=100, h=100, name="a.jpg", color=(1, 2, 3)):
        self.image = Image.new("RGB", (w, h), color)
        self.width = w
        self.height = h
        self.path = Path(name)


@pytest.fixture
def patched(monkeypatch):
    """Remplace ingest/caption/panorama par des doublures qui enregistrent."""
    calls = {}

    def fake_ingest(input_dir, *, min_images=3):
        calls["ingest"] = {"input_dir": input_dir, "min_images": min_images}
        return [
            FakeImg(800, 600, "a.jpg"),
            FakeImg(1920, 1080, "b.jpg"),  # plus haute résolution
            FakeImg(640, 480, "c.jpg"),
        ]

    def fake_describe(source, *, client=None, extra_guidance=None, hf_token=None):
        calls["describe"] = {"source": source}
        return "a cozy 360 living room"

    def fake_generate(prompt, *, seed=0, num_inference_steps=50):
        calls["generate"] = {"prompt": prompt, "seed": seed, "steps": num_inference_steps}
        return Panorama(
            image=Image.new("RGB", (2048, 1024)),
            width=2048,
            height=1024,
            prompt=prompt,
            backend="hf_space",
        )

    monkeypatch.setattr(forge_module, "ingest", fake_ingest)
    monkeypatch.setattr(forge_module, "describe_room", fake_describe)
    monkeypatch.setattr(forge_module, "generate_panorama", fake_generate)
    return calls


# --- sélection d'images -----------------------------------------------------


def test_select_images_picks_highest_resolution():
    imgs = [FakeImg(800, 600, "a.jpg"), FakeImg(1920, 1080, "b.jpg")]
    assert _select_images(imgs, multi=False)[0].path.name == "b.jpg"


def test_select_images_multi_limit():
    imgs = [FakeImg(100, 100, f"{i}.jpg") for i in range(6)]
    assert len(_select_images(imgs, multi=True)) == MULTI_IMAGE_LIMIT


def test_select_images_multi_maximizes_diversity():
    # Deux rouges quasi identiques + 3 couleurs distinctes : la sélection doit
    # prendre les vues distinctes et écarter l'un des doublons rouges.
    imgs = [
        FakeImg(100, 100, "red.jpg", (255, 0, 0)),
        FakeImg(100, 100, "red2.jpg", (250, 5, 5)),
        FakeImg(100, 100, "green.jpg", (0, 255, 0)),
        FakeImg(100, 100, "blue.jpg", (0, 0, 255)),
        FakeImg(100, 100, "yellow.jpg", (255, 255, 0)),
    ]
    chosen = {im.path.name for im in _select_images(imgs, multi=True)}
    assert {"green.jpg", "blue.jpg", "yellow.jpg"} <= chosen
    assert len({"red.jpg", "red2.jpg"} & chosen) == 1  # un seul des deux rouges


# --- backend worldlabs (défaut) ---------------------------------------------


def test_forge_worldlabs_exports_real_photo(tmp_path, patched):
    result = forge(tmp_path / "input", tmp_path / "out")

    assert result.backend == "worldlabs"
    assert result.prompt is None  # pas de caption en worldlabs
    assert result.panorama is None
    assert len(result.source_images) == 1
    assert result.source_images[0].name == "source-00.jpg"
    assert result.source_images[0].exists()
    # chaque run écrit dans un sous-dossier dédié output/<run>/
    assert result.run_dir.parent == tmp_path / "out"
    assert result.source_images[0].parent == result.run_dir
    manifest = json.loads(
        (result.run_dir / "world_labs_request.json").read_text(encoding="utf-8")
    )
    assert manifest["backend"] == "worldlabs"
    assert manifest["multi"] is False


def test_forge_worldlabs_multi_exports_four(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forge_module,
        "ingest",
        lambda d, *, min_images=3: [FakeImg(100, 100, f"{i}.jpg") for i in range(5)],
    )
    result = forge(tmp_path / "input", tmp_path / "out", multi=True)
    assert len(result.source_images) == MULTI_IMAGE_LIMIT
    assert result.world_labs_request["multi"] is True


# --- backend dit360 ---------------------------------------------------------


def test_forge_run_id_subdir(tmp_path, patched):
    result = forge(tmp_path / "input", tmp_path / "out", run_id="run-A")
    assert result.run_dir == tmp_path / "out" / "run-A"
    assert (tmp_path / "out" / "run-A" / "world_labs_request.json").exists()


def test_forge_dit360_uses_caption_and_panorama(tmp_path, patched):
    result = forge(tmp_path / "input", tmp_path / "out", backend="dit360")

    assert result.backend == "dit360"
    assert result.prompt == "a cozy 360 living room"
    assert result.panorama is not None
    assert result.source_images[0].name == "panorama.jpg"
    assert patched["generate"]["prompt"] == "a cozy 360 living room"


def test_forge_dit360_passes_seed(tmp_path, patched):
    forge(tmp_path / "input", tmp_path / "out", backend="dit360", seed=42)
    assert patched["generate"]["seed"] == 42


def test_forge_unknown_backend(tmp_path, patched):
    with pytest.raises(ValueError, match="Backend inconnu"):
        forge(tmp_path / "input", tmp_path / "out", backend="imagine")


# --- World Labs : construction de requête ------------------------------------


def test_build_world_request_single_base64(tmp_path):
    pano = tmp_path / "panorama.jpg"
    Image.new("RGB", (8, 8)).save(pano)

    req = _build_world_request(
        [(pano, 0)],
        prompt="a room",
        display_name="demo",
        multi=False,
        api_key="k",
        transport=None,
        upload=None,
    )
    assert req["model"] == WORLD_LABS_MODEL
    ip = req["world_prompt"]["image_prompt"]
    assert ip["source"] == "data_base64" and ip["data_base64"]
    assert req["world_prompt"]["text_prompt"] == "a room"


def test_build_world_request_multi_uploads(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"s{i}.jpg"
        Image.new("RGB", (8, 8)).save(p)
        paths.append((p, i * 180))

    uploaded = []

    def transport(url, *, api_key, method="GET", payload=None):
        return {
            "media_asset": {"media_asset_id": f"asset-{len(uploaded)}"},
            "upload_info": {"upload_url": "http://up/x", "upload_method": "PUT"},
        }

    def upload(url, path, *, method="PUT", headers=None):
        uploaded.append((url, str(path)))

    req = _build_world_request(
        paths,
        prompt=None,
        display_name="demo",
        multi=True,
        api_key="k",
        transport=transport,
        upload=upload,
    )
    assert req["world_prompt"]["type"] == "multi-image"
    entries = req["world_prompt"]["multi_image_prompt"]
    assert len(entries) == 2
    assert entries[0]["content"]["source"] == "media_asset"
    assert entries[1]["azimuth"] == 180
    assert len(uploaded) == 2


def test_build_video_request_uploads_as_video(tmp_path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake video bytes")
    captured = {}

    def transport(url, *, api_key, method="GET", payload=None):
        captured["kind"] = payload["kind"]
        captured["extension"] = payload["extension"]
        return {
            "media_asset": {"media_asset_id": "vid-1"},
            "upload_info": {"upload_url": "http://up/x"},
        }

    def upload(url, path, *, method="PUT", headers=None):
        captured["uploaded"] = str(path)

    req = _build_video_request(
        vid, prompt="a room", display_name="demo",
        api_key="k", transport=transport, upload=upload,
    )
    assert req["model"] == WORLD_LABS_MODEL
    assert req["world_prompt"]["type"] == "video"
    assert req["world_prompt"]["video_prompt"] == {
        "source": "media_asset",
        "media_asset_id": "vid-1",
    }
    assert req["world_prompt"]["text_prompt"] == "a room"
    assert captured["kind"] == "video"
    assert captured["extension"] == "mp4"


def test_submit_world_labs_video(tmp_path):
    vid = tmp_path / "clip.mov"
    vid.write_bytes(b"x")

    def transport(url, *, api_key, method="GET", payload=None):
        if url.endswith("prepare_upload"):
            return {
                "media_asset": {"media_asset_id": "v"},
                "upload_info": {"upload_url": "http://up/x"},
            }
        return {"operation_id": "o", "done": True, "response": {"assets": {"v": 1}}}

    def upload(url, path, *, method="PUT", headers=None):
        pass

    world = submit_world_labs(
        video=vid, api_key="k", poll_interval=0, _transport=transport, _upload=upload
    )
    assert world == {"assets": {"v": 1}}


def test_submit_world_labs_requires_input():
    with pytest.raises(WorldLabsError, match="images.*video|video"):
        submit_world_labs(api_key="k", _transport=lambda *a, **k: {})


def test_forge_video_skips_ingest(tmp_path, monkeypatch):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"x")
    captured = {}

    def fake_submit(images=None, *, video=None, prompt=None, display_name="pano-forge", multi=False):
        captured["video"] = video
        return {"assets": {}}

    def boom_ingest(*a, **k):
        raise AssertionError("ingest ne doit pas tourner en mode vidéo")

    monkeypatch.setattr(forge_module, "ingest", boom_ingest)
    monkeypatch.setattr(forge_module, "submit_world_labs", fake_submit)
    monkeypatch.setattr(forge_module, "download_world_assets", lambda w, o: {})

    result = forge(tmp_path / "input", tmp_path / "out", video=vid, submit=True)
    assert result.backend == "video"
    assert result.source_images == [vid]
    assert captured["video"] == vid
    manifest = json.loads(
        (result.run_dir / "world_labs_request.json").read_text(encoding="utf-8")
    )
    assert manifest["backend"] == "video"


def test_forge_video_missing_file(tmp_path):
    with pytest.raises(WorldLabsError, match="introuvable"):
        forge(tmp_path / "input", tmp_path / "out", video=tmp_path / "nope.mp4")


# --- mode mix : MP4 combiné (vidéo + photos en frames fixes) -> World Labs vidéo --


def test_forge_mix_builds_combined_video_and_submits(tmp_path, monkeypatch):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"x")
    captured = {}

    def fake_ingest(input_dir, *, min_images=3):
        return [FakeImg(100, 100, "p1.jpg"), FakeImg(100, 100, "p2.jpg")]

    def fake_build(
        video_path, photos, dest, *, still_seconds=5.0, max_frames=50,
        min_sharpness=50.0, min_motion=5.0, transition_frames=8, cache_dir=None, stats=None,
    ):
        captured["photos"] = len(photos)
        captured["still"] = still_seconds
        captured["max_frames"] = max_frames
        captured["min_sharpness"] = min_sharpness
        captured["min_motion"] = min_motion
        captured["transition_frames"] = transition_frames
        captured["cache_dir"] = cache_dir
        if stats is not None:
            stats.update({"frames_extracted": 5, "frames_retained": 3, "mp4_bytes": 3})
        Path(dest).write_bytes(b"mp4")
        return Path(dest)

    def fake_submit(images=None, *, video=None, prompt=None, display_name="pano-forge", multi=False):
        captured["video"] = video
        captured["multi"] = multi
        return {"assets": {}}

    monkeypatch.setattr(forge_module, "ingest", fake_ingest)
    monkeypatch.setattr(forge_module, "validate_video", lambda p: {"duration": 10.0})
    monkeypatch.setattr(forge_module, "_build_mix_video", fake_build)
    monkeypatch.setattr(forge_module, "submit_world_labs", fake_submit)
    monkeypatch.setattr(forge_module, "download_world_assets", lambda w, o: {})

    result = forge(
        tmp_path / "input", tmp_path / "out",
        mix=True, video=vid, still_seconds=3.0, frames=30, min_sharpness=75.0,
        min_motion=8.0, transition_frames=4, submit=True,
    )

    assert result.backend == "mix"
    # un seul MP4 combiné, envoyé en VIDÉO (pas multi-image)
    assert result.source_images == [result.run_dir / "mixed.mp4"]
    assert captured["video"] == result.run_dir / "mixed.mp4"
    assert "multi" not in captured or captured["multi"] is False
    assert captured["photos"] == 2  # les 2 photos ajoutées
    assert captured["still"] == 3.0
    assert captured["max_frames"] == 30  # --frames transmis à la construction
    assert captured["min_sharpness"] == 75.0  # --min-sharpness transmis
    assert captured["min_motion"] == 8.0  # --min-motion transmis
    assert captured["transition_frames"] == 4  # --transition-frames transmis
    assert captured["cache_dir"] == tmp_path / "out" / ".cache"  # cache partagé
    assert result.world_labs_request["backend"] == "mix"
    assert result.world_labs_request["photos"] == 2
    assert result.world_labs_request["still_seconds"] == 3.0
    assert result.world_labs_request["frames"] == 30
    assert result.world_labs_request["min_sharpness"] == 75.0
    # log structuré : métriques du build_mix + résultat World Labs
    run_log = json.loads((result.run_dir / "run.log.json").read_text(encoding="utf-8"))
    assert run_log["backend"] == "mix"
    assert run_log["mix"]["frames_extracted"] == 5
    assert "build_mix" in run_log["steps"] and "validate" in run_log["steps"]
    assert run_log["world"]["status"] == "done"


def test_forge_mix_requires_video(tmp_path):
    with pytest.raises(ValueError, match="--mix nécessite --video"):
        forge(tmp_path / "input", tmp_path / "out", mix=True)


def test_forge_mix_tolerates_no_photos(tmp_path, monkeypatch):
    from src.ingest import IngestError

    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"x")
    captured = {}

    def boom_ingest(*a, **k):
        raise IngestError("aucune photo")

    def fake_build(
        video_path, photos, dest, *, still_seconds=5.0, max_frames=50,
        min_sharpness=50.0, min_motion=5.0, transition_frames=8, cache_dir=None, stats=None,
    ):
        captured["photos"] = len(photos)
        Path(dest).write_bytes(b"mp4")
        return Path(dest)

    monkeypatch.setattr(forge_module, "ingest", boom_ingest)
    monkeypatch.setattr(forge_module, "validate_video", lambda p: {"duration": 10.0})
    monkeypatch.setattr(forge_module, "_build_mix_video", fake_build)

    result = forge(tmp_path / "input", tmp_path / "out", mix=True, video=vid)
    assert result.backend == "mix"
    assert captured["photos"] == 0  # aucune photo -> juste la vidéo


def test_build_mix_video_keeps_motion_frames_and_interleaves(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(0)
    src = tmp_path / "src.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(src), fourcc, 10.0, (48, 48))
    if not writer.isOpened():
        pytest.skip("encodeur mp4v indisponible")
    # Bruit aléatoire par frame : fort mouvement ET forte netteté (Laplacien élevé).
    for _ in range(10):
        writer.write(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8))
    writer.release()

    photos = [Image.fromarray(rng.integers(0, 256, (32, 24, 3), dtype=np.uint8))]
    out = forge_module._build_mix_video(
        src, photos, tmp_path / "mixed.mp4", still_seconds=0.5, max_frames=3
    )
    assert out.exists() and out.stat().st_size > 0

    # Le mix relit bien une vidéo valide (keyframes + photo intercalée).
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()


def test_build_mix_video_falls_back_when_all_blurry(tmp_path):
    # Frames/photo uniformes (netteté nulle) : le garde-fou évite un MP4 vide.
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    src = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("encodeur mp4v indisponible")
    for i in range(5):
        writer.write(np.full((32, 32, 3), i * 20, dtype=np.uint8))
    writer.release()

    out = forge_module._build_mix_video(
        src, [], tmp_path / "mixed.mp4", still_seconds=0.5, max_frames=3
    )
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0  # frames gardées malgré le flou
    cap.release()


# --- mix : détection de mouvement / intercalage / flou / netteté / luminosité --


def test_select_motion_frames_keeps_top_movers_in_order():
    # scores : la frame 2 et 4 bougent le plus ; on garde 2 keyframes, en ordre.
    scores = [0.0, 1.0, 9.0, 2.0, 8.0]
    assert _select_motion_frames(scores, 2) == [2, 4]


def test_select_motion_frames_caps_to_available():
    assert _select_motion_frames([0.0, 1.0], 50) == [0, 1]
    assert _select_motion_frames([], 50) == []


def test_select_motion_frames_restricts_to_eligible():
    # Même si la frame 2 bouge le plus, elle est exclue si pas dans `eligible`.
    scores = [0.0, 1.0, 9.0, 2.0, 8.0]
    assert _select_motion_frames(scores, 2, eligible=[0, 3, 4]) == [3, 4]


def test_sharpness_higher_for_edges_than_flat():
    pytest.importorskip("cv2")
    import numpy as np

    flat = np.full((40, 40, 3), 120, dtype=np.uint8)
    edged = flat.copy()
    edged[:, 20:] = 255  # bord net
    assert _sharpness(edged) > _sharpness(flat)
    assert _sharpness(flat) == 0.0


def test_auto_brightness_lifts_dark_image_only():
    pytest.importorskip("cv2")
    import numpy as np

    # Image sombre (luma ~30) : doit être éclaircie.
    dark = np.full((32, 32, 3), 30, dtype=np.uint8)
    dark[:16] = 20  # un peu de variation pour que equalizeHist agisse
    lifted = _auto_brightness_bgr(dark)
    assert forge_module._mean_luma(lifted) > forge_module._mean_luma(dark)

    # Image déjà claire (luma ~200) : inchangée (au-dessus du seuil).
    bright = np.full((32, 32, 3), 200, dtype=np.uint8)
    assert np.array_equal(_auto_brightness_bgr(bright), bright)


def test_upscale_if_low_res_only_upscales_small():
    pytest.importorskip("cv2")
    import numpy as np

    small = np.zeros((240, 320, 3), dtype=np.uint8)  # côté court 240 < 720
    up = _upscale_if_low_res(small)
    assert min(up.shape[:2]) >= 720

    big = np.zeros((1080, 1920, 3), dtype=np.uint8)  # déjà >= 720 : inchangé
    assert _upscale_if_low_res(big).shape == big.shape


def test_rotate_bgr_swaps_dimensions_for_90():
    pytest.importorskip("cv2")
    import numpy as np

    frame = np.zeros((40, 60, 3), dtype=np.uint8)  # h=40, w=60
    assert _rotate_bgr(frame, 90).shape[:2] == (60, 40)
    assert _rotate_bgr(frame, 180).shape[:2] == (40, 60)
    assert _rotate_bgr(frame, 0).shape[:2] == (40, 60)


def test_open_video_writer_picks_available_codec(tmp_path):
    cv2 = pytest.importorskip("cv2")

    dest = tmp_path / "out.mp4"
    writer, codec = _open_video_writer(dest, 10.0, 64, 64)
    try:
        assert codec in MIX_CODECS
        assert writer.isOpened()
    finally:
        writer.release()


def test_image_quality_metrics_distinguishes_sharp_and_blurry(tmp_path):
    pytest.importorskip("cv2")
    import numpy as np
    from PIL import ImageFilter

    rng = np.random.default_rng(0)
    sharp_img = Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
    sharp_img.save(tmp_path / "sharp.png")
    blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(6))
    blurry_img.save(tmp_path / "blurry.png")

    sharp = _image_quality_metrics(tmp_path / "sharp.png")
    blurry = _image_quality_metrics(tmp_path / "blurry.png")

    assert set(sharp) == {"sharpness", "brightness", "contrast", "blur_ratio"}
    assert sharp["sharpness"] > blurry["sharpness"]
    assert blurry["blur_ratio"] >= sharp["blur_ratio"]
    assert 0.0 <= sharp["blur_ratio"] <= 1.0


def test_world_status_reports_states():
    assert _world_status(None, submit=False) == {"status": "not_submitted"}
    assert _world_status(None, submit=True) == {"status": "no_response"}
    status = _world_status({"assets": {"glb": 1, "pano": 2}}, submit=True)
    assert status["status"] == "done"
    assert status["assets"] == ["glb", "pano"]


def test_forge_worldlabs_writes_run_log(tmp_path, patched):
    # Même sans --submit, un run.log.json structuré est écrit.
    result = forge(tmp_path / "input", tmp_path / "out")
    run_log = json.loads((result.run_dir / "run.log.json").read_text(encoding="utf-8"))
    assert run_log["backend"] == "worldlabs"
    assert run_log["world"]["status"] == "not_submitted"
    assert run_log["source_images"] == 1


def test_assign_photos_to_keyframes_picks_most_similar():
    import numpy as np

    key_feats = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    photo_feats = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
    assert _assign_photos_to_keyframes(photo_feats, key_feats) == {0: [0], 1: [1]}


def test_assign_photos_to_keyframes_empty_keyframes():
    import numpy as np

    assert _assign_photos_to_keyframes([np.array([1.0])], []) == {}


def test_motion_blur_preserves_shape_and_softens():
    pytest.importorskip("cv2")
    import numpy as np

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, 20:] = 255  # bord net vertical
    blurred = _motion_blur_bgr(img)
    assert blurred.shape == img.shape
    # Le flou horizontal crée des valeurs intermédiaires à la frontière nette.
    assert blurred[:, 18:22].std() > 0
    assert not np.array_equal(blurred, img)


def test_operation_id_extracts_last_segment():
    assert _operation_id({"operation_id": "orgs/x/operations/abc123"}) == "abc123"
    with pytest.raises(WorldLabsError, match="operation_id"):
        _operation_id({})


# --- World Labs : submit + poll ---------------------------------------------


def test_submit_world_labs_single_then_polls(tmp_path):
    pano = tmp_path / "panorama.jpg"
    Image.new("RGB", (8, 8)).save(pano)

    responses = iter([
        {"operation_id": "ops/abc", "done": False},
        {"operation_id": "ops/abc", "done": False},
        {"operation_id": "ops/abc", "done": True, "response": {"assets": {}}},
    ])
    calls = []

    def transport(url, *, api_key, method="GET", payload=None):
        calls.append((method, url))
        return next(responses)

    world = submit_world_labs(
        pano, prompt="a room", api_key="k", poll_interval=0, _transport=transport
    )
    assert world == {"assets": {}}
    assert calls[0] == ("POST", f"{forge_module.WORLD_LABS_ENDPOINT}/worlds:generate")
    assert calls[1][1].endswith("/operations/abc")


def test_submit_world_labs_multi(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"s{i}.jpg"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    uploaded = []

    def transport(url, *, api_key, method="GET", payload=None):
        if url.endswith("prepare_upload"):
            return {
                "media_asset": {"media_asset_id": f"asset-{len(uploaded)}"},
                "upload_info": {"upload_url": "http://up/x"},
            }
        # worlds:generate -> terminé d'emblée
        return {"operation_id": "o", "done": True, "response": {"assets": {"x": 1}}}

    def upload(url, path, *, method="PUT", headers=None):
        uploaded.append(str(path))

    world = submit_world_labs(
        paths, api_key="k", multi=True, poll_interval=0,
        _transport=transport, _upload=upload,
    )
    assert world == {"assets": {"x": 1}}
    assert len(uploaded) == 2


def test_submit_world_labs_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_LABS_API_KEY", raising=False)
    pano = tmp_path / "panorama.jpg"
    Image.new("RGB", (8, 8)).save(pano)
    with pytest.raises(WorldLabsError, match="WORLD_LABS_API_KEY"):
        submit_world_labs(pano, api_key=None, _transport=lambda *a, **k: {})


def test_submit_world_labs_propagates_error(tmp_path):
    pano = tmp_path / "panorama.jpg"
    Image.new("RGB", (8, 8)).save(pano)
    op = {"operation_id": "o", "done": True, "error": "bad input"}
    with pytest.raises(WorldLabsError, match="échou"):
        submit_world_labs(pano, api_key="k", poll_interval=0, _transport=lambda *a, **k: op)


# --- World Labs : retry sur erreurs serveur transitoires (500/503) -----------


class _FakeResp:
    def __init__(self, body=b"{}"):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _http_error(url, code, body=b"boom"):
    import io
    import urllib.error

    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(body))


def test_request_json_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req):
        calls["n"] += 1
        if calls["n"] <= 2:  # deux 500 transitoires, puis succès (cas vécu)
            raise _http_error(req.full_url, 500)
        return _FakeResp(b'{"ok": true}')

    monkeypatch.setattr(forge_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(forge_module.time, "sleep", lambda s: None)

    assert _request_json("http://x", api_key="k") == {"ok": True}
    assert calls["n"] == 3  # 2 échecs + 1 succès


def test_request_json_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req):
        calls["n"] += 1
        raise _http_error(req.full_url, 503)  # 503 persistant

    monkeypatch.setattr(forge_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(forge_module.time, "sleep", lambda s: None)

    with pytest.raises(WorldLabsError, match="503"):
        _request_json("http://x", api_key="k")
    assert calls["n"] == forge_module.WORLD_LABS_MAX_RETRIES + 1  # 1 essai + 3 retries


def test_request_json_no_retry_on_client_error(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req):
        calls["n"] += 1
        raise _http_error(req.full_url, 400, b"bad request")

    monkeypatch.setattr(forge_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(forge_module.time, "sleep", lambda s: None)

    with pytest.raises(WorldLabsError, match="400"):
        _request_json("http://x", api_key="k")
    assert calls["n"] == 1  # 4xx : pas de retry


# --- téléchargement des assets ----------------------------------------------


def test_download_world_assets(tmp_path, monkeypatch):
    world = {
        "assets": {
            "mesh": {"collider_mesh_url": "http://x/a.glb"},
            "imagery": {"pano_url": "http://x/p.png"},
            "thumbnail_url": "http://x/t.webp",
            "splats": {"spz_urls": {"100k": "http://x/s100.spz", "full_res": "http://x/sf.spz"}},
        }
    }
    seen = []

    def fake_dl(url, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        seen.append(url)
        return dest

    monkeypatch.setattr(forge_module, "_download_file", fake_dl)
    res = download_world_assets(world, tmp_path)

    assert res["glb"].name == "world.glb"
    assert res["pano"].name == "world-pano.png"
    assert res["thumbnail"].name == "world-thumbnail.webp"
    assert set(res["spz"]) == {"100k", "full_res"}
    assert len(seen) == 5


def test_download_world_assets_handles_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(forge_module, "_download_file", lambda url, dest: dest)
    assert download_world_assets({"assets": {}}, tmp_path) == {"spz": {}}


# --- forge --submit + CLI ---------------------------------------------------


def test_forge_submit_calls_world_labs(tmp_path, patched, monkeypatch):
    captured = {}

    def fake_submit(arg, *, prompt=None, display_name="pano-forge", multi=False):
        captured["arg"] = arg
        captured["multi"] = multi
        return {"assets": {"imagery": {"pano_url": "http://x/p.png"}}}

    monkeypatch.setattr(forge_module, "submit_world_labs", fake_submit)
    monkeypatch.setattr(
        forge_module, "download_world_assets", lambda world, out: {"pano": out / "p"}
    )
    result = forge(tmp_path / "input", tmp_path / "out", submit=True)

    assert result.world == {"assets": {"imagery": {"pano_url": "http://x/p.png"}}}
    assert result.world_assets == {"pano": result.run_dir / "p"}
    # single-image : l'argument est un chemin, pas une liste
    assert isinstance(captured["arg"], Path)
    assert captured["multi"] is False


def test_forge_submit_multi_passes_list(tmp_path, patched, monkeypatch):
    captured = {}

    def fake_submit(arg, *, prompt=None, display_name="pano-forge", multi=False):
        captured["arg"] = arg
        captured["multi"] = multi
        return {"assets": {}}

    monkeypatch.setattr(forge_module, "submit_world_labs", fake_submit)
    monkeypatch.setattr(forge_module, "download_world_assets", lambda world, out: {})
    forge(tmp_path / "input", tmp_path / "out", multi=True, submit=True)

    assert captured["multi"] is True
    assert isinstance(captured["arg"], list)  # liste de (path, azimuth)


def test_main_success(tmp_path, patched, capsys):
    rc = forge_module.main([str(tmp_path / "input"), "-o", str(tmp_path / "out")])
    assert rc == 0
    assert "Backend : worldlabs" in capsys.readouterr().out


def test_main_handles_pipeline_error(tmp_path, monkeypatch, capsys):
    from src.ingest import IngestError

    def boom(input_dir, *, min_images=3):
        raise IngestError("pas assez d'images")

    monkeypatch.setattr(forge_module, "ingest", boom)
    rc = forge_module.main([str(tmp_path / "input")])
    assert rc == 1
    assert "Erreur" in capsys.readouterr().out
