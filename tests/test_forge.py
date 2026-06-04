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
    PipelineResult,
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


def test_submit_world_labs_not_implemented():
    with pytest.raises(NotImplementedError):
        submit_world_labs({"endpoint": "x"})


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
