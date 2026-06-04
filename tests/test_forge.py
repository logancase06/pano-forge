"""Tests pour le pipeline complet (:mod:`src.forge`).

Les étapes réseau (ingest/caption/panorama) sont remplacées via monkeypatch :
on teste l'orchestration et le stub World Labs, pas les services externes.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from src import forge as forge_module
from src.forge import (
    WORLD_LABS_MODEL,
    PipelineResult,
    WorldLabsError,
    _build_world_request,
    _operation_id,
    forge,
    prepare_world_labs,
    submit_world_labs,
)
from src.panorama import Panorama


def _fake_panorama(prompt="a 360 room"):
    img = Image.new("RGB", (2048, 1024), (12, 34, 56))
    return Panorama(
        image=img, width=2048, height=1024, prompt=prompt, backend="hf_space"
    )


@pytest.fixture
def patched(monkeypatch):
    """Remplace ingest/caption/panorama par des doublures qui enregistrent."""
    calls = {}

    class FakeImg:
        image = Image.new("RGB", (10, 10))

    def fake_ingest(input_dir, *, min_images=3):
        calls["ingest"] = {"input_dir": input_dir, "min_images": min_images}
        return [FakeImg(), FakeImg(), FakeImg()]

    def fake_describe(source, *, client=None, extra_guidance=None):
        calls["describe"] = {"source": source, "extra_guidance": extra_guidance}
        return "a cozy 360 living room"

    def fake_generate(prompt, *, backend="hf_space", seed=0, num_inference_steps=50):
        calls["generate"] = {
            "prompt": prompt,
            "backend": backend,
            "seed": seed,
            "steps": num_inference_steps,
        }
        return _fake_panorama(prompt)

    monkeypatch.setattr(forge_module, "ingest", fake_ingest)
    monkeypatch.setattr(forge_module, "describe_room", fake_describe)
    monkeypatch.setattr(forge_module, "generate_panorama", fake_generate)
    return calls


def test_forge_runs_full_pipeline(tmp_path, patched):
    result = forge(tmp_path / "input", tmp_path / "out")

    assert isinstance(result, PipelineResult)
    assert result.prompt == "a cozy 360 living room"
    # Le prompt issu de la caption alimente bien la génération.
    assert patched["generate"]["prompt"] == "a cozy 360 living room"
    # Le panorama est sauvegardé.
    assert result.panorama_path.exists()
    assert result.panorama_path.name == "panorama.jpg"


def test_forge_passes_backend_and_seed(tmp_path, patched):
    forge(tmp_path / "input", tmp_path / "out", backend="local_dit360", seed=42)
    assert patched["generate"]["backend"] == "local_dit360"
    assert patched["generate"]["seed"] == 42


def test_forge_captions_first_image(tmp_path, patched):
    forge(tmp_path / "input", tmp_path / "out")
    # La source caption est la 1re image ingérée (un objet portant .image).
    assert hasattr(patched["describe"]["source"], "image")


def test_forge_writes_world_labs_manifest(tmp_path, patched):
    result = forge(tmp_path / "input", tmp_path / "out")
    manifest = tmp_path / "out" / "world_labs_request.json"

    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["input_type"] == "equirectangular_panorama"
    assert data["width"] == 2048 and data["height"] == 1024
    assert data["prompt"] == "a cozy 360 living room"
    assert result.world_labs_request == data


def test_prepare_world_labs_payload(tmp_path):
    pano = _fake_panorama("beach sunset")
    pano_path = tmp_path / "panorama.jpg"
    pano.save(pano_path)

    payload = prepare_world_labs(pano, pano_path, output_dir=tmp_path)
    assert payload["panorama_path"] == str(pano_path)
    assert payload["endpoint"].startswith("https://")


def test_build_world_request_has_base64_and_model(tmp_path):
    pano = tmp_path / "panorama.jpg"
    Image.new("RGB", (8, 8)).save(pano)

    req = _build_world_request(pano, prompt="a bright room", display_name="demo")
    assert req["model"] == WORLD_LABS_MODEL
    assert req["display_name"] == "demo"
    image_prompt = req["world_prompt"]["image_prompt"]
    assert image_prompt["source"] == "data_base64"
    assert image_prompt["data_base64"]  # non vide
    assert image_prompt["mime_type"] == "image/jpeg"
    assert req["world_prompt"]["text_prompt"] == "a bright room"


def test_operation_id_extracts_last_segment():
    assert _operation_id({"operation_id": "orgs/x/operations/abc123"}) == "abc123"
    assert _operation_id({"name": "op-42"}) == "op-42"
    with pytest.raises(WorldLabsError, match="operation_id"):
        _operation_id({})


def test_submit_world_labs_submits_then_polls(tmp_path):
    pano = tmp_path / "panorama.jpg"
    Image.new("RGB", (8, 8)).save(pano)

    responses = iter([
        {"operation_id": "ops/abc", "done": False},  # POST worlds:generate
        {"operation_id": "ops/abc", "done": False},  # 1er poll
        {
            "operation_id": "ops/abc",
            "done": True,
            "response": {"assets": {"imagery": {"pano_url": "http://x/p.png"}}},
        },
    ])
    calls = []

    def transport(url, *, api_key, method="GET", payload=None):
        calls.append((method, url, payload is not None))
        return next(responses)

    world = submit_world_labs(
        pano, prompt="a room", api_key="k", poll_interval=0, _transport=transport
    )

    assert world["assets"]["imagery"]["pano_url"] == "http://x/p.png"
    # 1er appel = POST worlds:generate avec corps ; ensuite GET operations/abc.
    assert calls[0] == ("POST", f"{forge_module.WORLD_LABS_ENDPOINT}/worlds:generate", True)
    assert calls[1][0] == "GET" and calls[1][1].endswith("/operations/abc")


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


def test_forge_submit_calls_world_labs(tmp_path, patched, monkeypatch):
    captured = {}

    def fake_submit(panorama_path, *, prompt=None, display_name="pano-forge"):
        captured["path"] = panorama_path
        captured["prompt"] = prompt
        return {"assets": {"imagery": {"pano_url": "http://x/p.png"}}}

    monkeypatch.setattr(forge_module, "submit_world_labs", fake_submit)
    monkeypatch.setattr(
        forge_module, "download_world_assets", lambda world, out: {"pano": out / "p"}
    )
    result = forge(tmp_path / "input", tmp_path / "out", submit=True)

    assert result.world == {"assets": {"imagery": {"pano_url": "http://x/p.png"}}}
    assert captured["prompt"] == "a cozy 360 living room"
    assert result.world_assets == {"pano": (tmp_path / "out") / "p"}


def test_download_world_assets(tmp_path, monkeypatch):
    world = {
        "assets": {
            "mesh": {"collider_mesh_url": "http://x/a.glb"},
            "imagery": {"pano_url": "http://x/p.png"},
            "thumbnail_url": "http://x/t.webp",
            "splats": {
                "spz_urls": {
                    "100k": "http://x/s100.spz",
                    "full_res": "http://x/sfull.spz",
                }
            },
        }
    }
    seen = []

    def fake_dl(url, dest):
        from pathlib import Path

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        seen.append(url)
        return dest

    monkeypatch.setattr(forge_module, "_download_file", fake_dl)
    res = forge_module.download_world_assets(world, tmp_path)

    assert res["glb"].name == "world.glb"
    assert res["pano"].name == "world-pano.png"
    assert res["thumbnail"].name == "world-thumbnail.webp"
    assert set(res["spz"]) == {"100k", "full_res"}
    assert res["spz"]["100k"].name == "world-100k.spz"
    assert len(seen) == 5


def test_download_world_assets_handles_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forge_module, "_download_file", lambda url, dest: dest
    )
    res = forge_module.download_world_assets({"assets": {}}, tmp_path)
    assert res == {"spz": {}}


def test_main_success(tmp_path, patched, capsys):
    rc = forge_module.main([str(tmp_path / "input"), "-o", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Panorama" in out


def test_main_handles_pipeline_error(tmp_path, monkeypatch, capsys):
    from src.ingest import IngestError

    def boom(input_dir, *, min_images=3):
        raise IngestError("pas assez d'images")

    monkeypatch.setattr(forge_module, "ingest", boom)
    rc = forge_module.main([str(tmp_path / "input")])
    assert rc == 1
    assert "Erreur" in capsys.readouterr().out
