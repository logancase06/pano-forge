"""Tests pour l'étape de caption (:mod:`src.caption`).

Aucun appel réseau : on injecte un faux client gradio.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.caption import (
    DESCRIBE_PROMPT,
    MAX_IMAGE_EDGE,
    CaptionError,
    _prepare_image_file,
    _to_text,
    describe_room,
)


class FakeClient:
    """Imite gradio_client.Client : enregistre l'appel et rend (texte, durée)."""

    def __init__(self, reply="a cozy scandinavian living room, 360 panorama"):
        self.reply = reply
        self.calls = []

    def predict(self, *args, api_name=None, fn_index=None):
        self.calls.append({"args": args, "api_name": api_name, "fn_index": fn_index})
        return (self.reply, "1.23s")


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "room.jpg"
    Image.new("RGB", (64, 48), (200, 180, 150)).save(path)
    return path


def test_describe_room_from_path(photo):
    client = FakeClient()
    prompt = describe_room(photo, client=client)

    assert prompt == "a cozy scandinavian living room, 360 panorama"
    assert len(client.calls) == 1


def test_describe_room_sends_image_and_prompt(photo):
    client = FakeClient()
    describe_room(photo, client=client)
    args = client.calls[0]["args"]

    # (image, prompt) — le 2e argument est la consigne de description.
    assert args[1] == DESCRIBE_PROMPT


def test_extra_guidance_appended(photo):
    client = FakeClient()
    describe_room(photo, client=client, extra_guidance="Style: minimalist.")
    prompt_arg = client.calls[0]["args"][1]
    assert "minimalist" in prompt_arg


def test_describe_room_from_pil_image():
    client = FakeClient()
    assert describe_room(Image.new("RGB", (32, 32), (10, 20, 30)), client=client)


def test_describe_room_from_ingested_image(photo):
    client = FakeClient()

    class FakeIngested:
        def __init__(self, image):
            self.image = image

    ingested = FakeIngested(Image.open(photo).convert("RGB"))
    assert describe_room(ingested, client=client)


def test_falls_back_to_fn_index(photo):
    class PickyClient(FakeClient):
        def predict(self, *args, api_name=None, fn_index=None):
            if api_name is not None:
                raise ValueError("pas d'endpoint nommé")
            return super().predict(*args, fn_index=fn_index)

    client = PickyClient()
    assert describe_room(photo, client=client)
    assert client.calls[-1]["fn_index"] == 0


def test_temp_file_is_cleaned_up(photo, monkeypatch):
    seen = {}

    class CapturingClient(FakeClient):
        def predict(self, *args, api_name=None, fn_index=None):
            # Le 1er arg référence le fichier temporaire (chemin ou handle_file).
            seen["arg"] = args[0]
            return super().predict(*args, api_name=api_name, fn_index=fn_index)

    describe_room(photo, client=CapturingClient())
    # Le fichier temporaire ne doit plus exister après l'appel.
    arg = seen["arg"]
    path = arg.get("path") if isinstance(arg, dict) else arg
    from pathlib import Path

    assert not Path(path).exists()


def test_corrupt_image_raises(tmp_path):
    bad = tmp_path / "room.jpg"
    bad.write_bytes(b"not really a jpeg")
    with pytest.raises(CaptionError, match="illisible|introuvable"):
        describe_room(bad, client=FakeClient())


def test_large_image_is_downscaled():
    big = Image.new("RGB", (4000, 3000), (1, 2, 3))
    path = _prepare_image_file(big)
    try:
        with Image.open(path) as decoded:
            size = decoded.size
        assert max(size) == MAX_IMAGE_EDGE
    finally:
        path.unlink(missing_ok=True)


def test_default_client_passes_hf_token(monkeypatch):
    import gradio_client

    import src.caption as cap

    created = {}

    class FakeClient:
        def __init__(self, space_id, token=None, **kw):
            created["space_id"] = space_id
            created["token"] = token

    monkeypatch.setattr(gradio_client, "Client", FakeClient)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    cap._default_client("some/space")
    assert created["token"] == "hf_secret"


def test_default_client_no_token(monkeypatch):
    import gradio_client

    import src.caption as cap

    created = {}

    class FakeClient:
        def __init__(self, space_id, token=None, **kw):
            created["token"] = token

    monkeypatch.setattr(gradio_client, "Client", FakeClient)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cap._default_client("some/space")
    assert created["token"] is None


def test_to_text_handles_tuple():
    assert _to_text(("hello world", "0.5s")) == "hello world"


def test_to_text_empty_raises():
    with pytest.raises(CaptionError, match="ne contient pas de texte"):
        _to_text(("   ", "0.1s"))
