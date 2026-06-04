"""Tests pour l'étape de génération panorama (:mod:`src.panorama`).

Les appels réseau (Space Gradio) et les dépendances lourdes (diffusers/torch)
ne sont pas requis : on injecte un faux client Gradio.
"""

from __future__ import annotations

import pytest
from PIL import Image

from src.panorama import (
    DEFAULT_STEPS,
    HFSpaceBackend,
    LocalDiT360Backend,
    Panorama,
    PanoramaError,
    build_backend,
    generate_panorama,
    _to_pil,
)


class FakeClient:
    """Imite gradio_client.Client : enregistre l'appel et rend un chemin image."""

    def __init__(self, space_id, image_path):
        self.space_id = space_id
        self._image_path = image_path
        self.calls = []

    def predict(self, *args, api_name=None, fn_index=None):
        self.calls.append({"args": args, "api_name": api_name, "fn_index": fn_index})
        return str(self._image_path)


@pytest.fixture
def pano_file(tmp_path):
    path = tmp_path / "pano.jpg"
    Image.new("RGB", (2048, 1024), (10, 20, 30)).save(path)
    return path


def test_build_backend_default_is_space():
    assert isinstance(build_backend(), HFSpaceBackend)


def test_build_backend_local():
    assert isinstance(build_backend("local_dit360"), LocalDiT360Backend)


def test_build_backend_unknown():
    with pytest.raises(PanoramaError, match="Backend inconnu"):
        build_backend("nope")


def test_space_backend_generates_panorama(pano_file):
    captured = {}

    def factory(space_id):
        client = FakeClient(space_id, pano_file)
        captured["client"] = client
        return client

    backend = HFSpaceBackend(client_factory=factory)
    pano = backend.generate("un salon scandinave", seed=7, num_inference_steps=40)

    assert isinstance(pano, Panorama)
    assert (pano.width, pano.height) == (2048, 1024)
    assert pano.is_equirectangular
    assert pano.backend == "hf_space"
    # Les arguments sont passés dans le bon ordre au Space.
    assert captured["client"].calls[0]["args"] == ("un salon scandinave", 7, 40)


def test_space_backend_rejects_empty_prompt(pano_file):
    backend = HFSpaceBackend(client_factory=lambda s: FakeClient(s, pano_file))
    with pytest.raises(PanoramaError, match="prompt"):
        backend.generate("   ")


def test_space_backend_falls_back_to_fn_index(pano_file):
    class PickyClient(FakeClient):
        def predict(self, *args, api_name=None, fn_index=None):
            if api_name is not None:
                raise ValueError("pas d'endpoint nommé")
            return super().predict(*args, fn_index=fn_index)

    client = PickyClient("space", pano_file)
    backend = HFSpaceBackend(client_factory=lambda s: client)
    pano = backend.generate("test")

    assert pano.is_equirectangular
    assert client.calls[-1]["fn_index"] == 0


def test_generate_panorama_with_backend_instance(pano_file):
    backend = HFSpaceBackend(client_factory=lambda s: FakeClient(s, pano_file))
    pano = generate_panorama("plage au coucher du soleil", backend=backend)
    assert pano.prompt == "plage au coucher du soleil"


def test_generate_panorama_uses_default_steps(pano_file):
    captured = {}

    def factory(space_id):
        client = FakeClient(space_id, pano_file)
        captured["client"] = client
        return client

    backend = HFSpaceBackend(client_factory=factory)
    generate_panorama("x", backend=backend)
    assert captured["client"].calls[0]["args"][2] == DEFAULT_STEPS


def test_local_backend_image_to_pano_not_implemented():
    backend = LocalDiT360Backend()
    with pytest.raises(NotImplementedError):
        backend.generate_from_image(Image.new("RGB", (10, 10)))


def test_panorama_save(tmp_path, pano_file):
    pano = Panorama(
        image=Image.open(pano_file).convert("RGB"),
        width=2048,
        height=1024,
        prompt="x",
        backend="hf_space",
    )
    out = tmp_path / "sub" / "result.jpg"
    saved = pano.save(out)
    assert saved.exists()


def test_to_pil_handles_tuple(pano_file):
    img = _to_pil((str(pano_file), "autre sortie ignorée"))
    assert isinstance(img, Image.Image)


def test_to_pil_handles_dict(pano_file):
    img = _to_pil({"path": str(pano_file)})
    assert isinstance(img, Image.Image)


def test_to_pil_rejects_garbage():
    with pytest.raises(PanoramaError):
        _to_pil(12345)
