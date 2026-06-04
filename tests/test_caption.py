"""Tests pour l'étape de caption (:mod:`src.caption`).

Aucun appel réseau ni clé API : on injecte un faux client Anthropic.
"""

from __future__ import annotations

import base64

import pytest
from PIL import Image

from src.caption import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    CaptionError,
    describe_room,
)


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class Resp:
            content = [FakeBlock(self.reply)]

        return Resp()


class FakeClient:
    def __init__(self, reply="a cozy scandinavian living room, 360 panorama"):
        self.messages = FakeMessages(reply)


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "room.jpg"
    Image.new("RGB", (64, 48), (200, 180, 150)).save(path)
    return path


def test_describe_room_from_path(photo):
    client = FakeClient()
    prompt = describe_room(photo, client=client)

    assert prompt == "a cozy scandinavian living room, 360 panorama"
    assert len(client.messages.calls) == 1


def test_uses_default_model_and_caches_system(photo):
    client = FakeClient()
    describe_room(photo, client=client)
    call = client.messages.calls[0]

    assert call["model"] == DEFAULT_MODEL
    # Le system est mis en cache (préfixe stable).
    assert call["system"][0]["text"] == SYSTEM_PROMPT
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_image_sent_as_base64(photo):
    client = FakeClient()
    describe_room(photo, client=client)
    content = client.messages.calls[0]["messages"][0]["content"]

    image_block = next(b for b in content if b["type"] == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"
    # Les données sont du base64 valide et non vide.
    assert base64.standard_b64decode(image_block["source"]["data"])


def test_describe_room_from_pil_image():
    client = FakeClient()
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    prompt = describe_room(img, client=client)
    assert prompt
    media_type = client.messages.calls[0]["messages"][0]["content"][0]["source"][
        "media_type"
    ]
    assert media_type == "image/jpeg"


def test_describe_room_from_ingested_image(photo):
    client = FakeClient()

    class FakeIngested:
        def __init__(self, image):
            self.image = image

    ingested = FakeIngested(Image.open(photo).convert("RGB"))
    assert describe_room(ingested, client=client)


def test_extra_guidance_appended(photo):
    client = FakeClient()
    describe_room(photo, client=client, extra_guidance="Style: minimaliste.")
    content = client.messages.calls[0]["messages"][0]["content"]
    text_block = next(b for b in content if b["type"] == "text")
    assert "minimaliste" in text_block["text"]


def test_unsupported_extension(tmp_path):
    bad = tmp_path / "room.tiff"
    bad.write_bytes(b"not really a tiff")
    with pytest.raises(CaptionError, match="non supporté"):
        describe_room(bad, client=FakeClient())


def test_empty_response_raises(photo):
    client = FakeClient(reply="   ")
    with pytest.raises(CaptionError, match="ne contient pas de texte"):
        describe_room(photo, client=client)


def test_sdk_error_wrapped(photo):
    class BoomClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("network down")

    with pytest.raises(CaptionError, match="API Anthropic"):
        describe_room(photo, client=BoomClient())
