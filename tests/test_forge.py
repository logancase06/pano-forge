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
    MULTI_IMAGE_LIMIT,
    WORLD_LABS_MODEL,
    PipelineResult,
    WorldLabsError,
    _assign_photos_to_keyframes,
    _build_video_request,
    _build_world_request,
    _motion_blur_bgr,
    _operation_id,
    _select_images,
    _select_motion_frames,
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

    def fake_build(video_path, photos, dest, *, still_seconds=5.0, max_frames=50):
        captured["photos"] = len(photos)
        captured["still"] = still_seconds
        captured["max_frames"] = max_frames
        Path(dest).write_bytes(b"mp4")
        return Path(dest)

    def fake_submit(images=None, *, video=None, prompt=None, display_name="pano-forge", multi=False):
        captured["video"] = video
        captured["multi"] = multi
        return {"assets": {}}

    monkeypatch.setattr(forge_module, "ingest", fake_ingest)
    monkeypatch.setattr(forge_module, "_build_mix_video", fake_build)
    monkeypatch.setattr(forge_module, "submit_world_labs", fake_submit)
    monkeypatch.setattr(forge_module, "download_world_assets", lambda w, o: {})

    result = forge(
        tmp_path / "input", tmp_path / "out",
        mix=True, video=vid, still_seconds=3.0, frames=30, submit=True,
    )

    assert result.backend == "mix"
    # un seul MP4 combiné, envoyé en VIDÉO (pas multi-image)
    assert result.source_images == [result.run_dir / "mixed.mp4"]
    assert captured["video"] == result.run_dir / "mixed.mp4"
    assert "multi" not in captured or captured["multi"] is False
    assert captured["photos"] == 2  # les 2 photos ajoutées
    assert captured["still"] == 3.0
    assert captured["max_frames"] == 30  # --frames transmis à la construction
    assert result.world_labs_request["backend"] == "mix"
    assert result.world_labs_request["photos"] == 2
    assert result.world_labs_request["still_seconds"] == 3.0
    assert result.world_labs_request["frames"] == 30


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

    def fake_build(video_path, photos, dest, *, still_seconds=5.0, max_frames=50):
        captured["photos"] = len(photos)
        Path(dest).write_bytes(b"mp4")
        return Path(dest)

    monkeypatch.setattr(forge_module, "ingest", boom_ingest)
    monkeypatch.setattr(forge_module, "_build_mix_video", fake_build)

    result = forge(tmp_path / "input", tmp_path / "out", mix=True, video=vid)
    assert result.backend == "mix"
    assert captured["photos"] == 0  # aucune photo -> juste la vidéo


def test_build_mix_video_keeps_motion_frames_and_interleaves(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    src = tmp_path / "src.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(src), fourcc, 10.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("encodeur mp4v indisponible")
    # 10 frames, niveaux de gris croissants : chaque frame diffère de la précédente.
    for i in range(10):
        writer.write(np.full((32, 32, 3), i * 25, dtype=np.uint8))
    writer.release()

    photos = [Image.new("RGB", (20, 16), (255, 0, 0))]
    out = forge_module._build_mix_video(
        src, photos, tmp_path / "mixed.mp4", still_seconds=0.5, max_frames=3
    )
    assert out.exists() and out.stat().st_size > 0

    # Le mix relit bien une vidéo valide (keyframes + photo intercalée).
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()


# --- mix : détection de mouvement / intercalage / flou ----------------------


def test_select_motion_frames_keeps_top_movers_in_order():
    # scores : la frame 2 et 4 bougent le plus ; on garde 2 keyframes, en ordre.
    scores = [0.0, 1.0, 9.0, 2.0, 8.0]
    assert _select_motion_frames(scores, 2) == [2, 4]


def test_select_motion_frames_caps_to_available():
    assert _select_motion_frames([0.0, 1.0], 50) == [0, 1]
    assert _select_motion_frames([], 50) == []


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
