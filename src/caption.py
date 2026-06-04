"""Pont ingest → panorama : décrit une photo source en prompt texte riche.

Le Space DiT360 est texte→panorama : il génère un équirectangulaire à partir
d'un prompt. Pour partir d'une *photo* d'intérieur, on la décrit d'abord avec
un VLM (Qwen-VL) hébergé sur un **Space Hugging Face public**, appelé via
``gradio_client`` — sans aucune clé API. Ce prompt alimente ensuite
:class:`src.panorama.HFSpaceBackend`.

Même pattern que :class:`src.panorama.HFSpaceBackend` : Space gratuit, sans GPU
local, ``client_factory`` injectable pour les tests.

⚠️ Les Spaces VLM exposent des API gradio variables (certains désactivent l'API,
d'autres utilisent un protocole de chat multi-étapes). ``space_id`` et
``api_name`` sont donc configurables, et l'appel retombe sur ``fn_index=0`` si
l'endpoint nommé n'existe pas. Le Space par défaut doit être validé en conditions
réelles ; pointe ``space_id`` vers un Space dont l'API gradio est ouverte.

Usage :

    from src.caption import describe_room
    from src.panorama import generate_panorama

    prompt = describe_room("input/salon.jpg")
    pano = generate_panorama(prompt)

L'import de ``gradio_client`` est paresseux : le module s'importe sans la
dépendance.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from PIL import Image

# Space VLM par défaut : Qwen2-VL-7B avec une API gradio ouverte, endpoint
# /run_example(image: filepath, text_input: str, model_id) -> output_text: str
# (vérifié via gradio_client.view_api). model_id a un défaut → on passe (image, prompt).
DEFAULT_SPACE_ID = "GanymedeNil/Qwen2-VL-7B"
# Endpoint nommé tenté en premier ; fallback sur fn_index=0 sinon.
DEFAULT_API_NAME = "/run_example"

# Bord long maximal envoyé : au-delà, on réduit (upload plus léger, moins lent).
MAX_IMAGE_EDGE = 1568

# Consigne unique (le Space ne distingue pas system/user) : produire un prompt
# 360° équirectangulaire pour DiT360 à partir de la photo.
DESCRIBE_PROMPT = (
    "You are shown a normal photo of an interior room. Write a single dense "
    "English prompt for an equirectangular 360 degree panorama generator "
    "(DiT360) that describes the ENTIRE room as a 360 scene, not just the "
    "photo's framing: wall layout, furniture and placement, floor and ceiling, "
    "materials and textures, color palette, lighting sources and mood, decor "
    "style, and overall atmosphere. Plausibly extrapolate the off-frame areas "
    "into a coherent, continuous 360 space. Reply ONLY with the descriptive "
    "prompt: no preamble, no quotes."
)


class CaptionError(Exception):
    """Erreur levée lorsque la génération du prompt échoue."""


def _load_image(source: str | Path | Image.Image | object) -> Image.Image:
    """Charge une source en image PIL RGB (chemin, image PIL, ou IngestedImage)."""
    # IngestedImage (ou tout objet portant une image PIL) → on prend son image.
    if not isinstance(source, (str, Path, Image.Image)) and hasattr(source, "image"):
        source = source.image

    if isinstance(source, Image.Image):
        img = source
    else:
        path = Path(source)
        try:
            img = Image.open(path)
            img.load()
        except FileNotFoundError as exc:
            raise CaptionError(f"Image introuvable : {path}") from exc
        except Exception as exc:  # noqa: BLE001 - UnidentifiedImageError, OSError…
            raise CaptionError(
                f"Image illisible ou corrompue : {path} ({exc})"
            ) from exc

    return img if img.mode == "RGB" else img.convert("RGB")


def _prepare_image_file(source: str | Path | Image.Image | object) -> Path:
    """Normalise la source en un fichier JPEG temporaire prêt pour ``gradio_client``.

    L'image est convertie en RGB et réduite si son bord long dépasse
    :data:`MAX_IMAGE_EDGE`. Le Space gradio attend un chemin de fichier (image),
    pas du base64. L'appelant est responsable de supprimer le fichier.
    """
    img = _load_image(source)

    longest = max(img.width, img.height)
    if longest > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / longest
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        img.save(tmp, format="JPEG", quality=90)
    finally:
        tmp.close()
    return Path(tmp.name)


def _to_text(result) -> str:
    """Normalise la sortie d'un appel gradio en texte.

    Le Space peut renvoyer une chaîne, un tuple ``(texte, durée)``, ou un dict.
    """
    if isinstance(result, (list, tuple)):
        if not result:
            raise CaptionError("Le Space n'a renvoyé aucune sortie.")
        result = result[0]
    if isinstance(result, dict):
        result = result.get("text") or result.get("value") or ""
    if not isinstance(result, str):
        result = str(result)

    text = result.strip()
    if not text:
        raise CaptionError("La réponse du VLM ne contient pas de texte.")
    return text


def _default_client(space_id: str, hf_token: str | None = None):
    """Construit un client gradio, en authentifiant si un token HF est dispo.

    Un token HF (argument ``hf_token`` ou variable d'environnement ``HF_TOKEN``)
    relève le quota ZeroGPU des Spaces ; il reste **optionnel**.
    """
    try:
        from gradio_client import Client
    except ImportError as exc:  # pragma: no cover - dépend de l'install
        raise CaptionError(
            "gradio_client est requis pour la caption. "
            "Installe-le : pip install gradio_client"
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        # Le nom du kwarg a changé : token (>=2.x) vs hf_token (1.x).
        try:
            return Client(space_id, token=token)
        except TypeError:
            return Client(space_id, hf_token=token)
    return Client(space_id)


def _as_file_arg(path: Path):
    """Enveloppe un chemin pour l'entrée image de gradio (handle_file si dispo)."""
    try:
        from gradio_client import handle_file

        return handle_file(str(path))
    except ImportError:  # gradio_client trop ancien : le chemin nu marche encore
        return str(path)


def describe_room(
    source: str | Path | Image.Image | object,
    *,
    client=None,
    space_id: str = DEFAULT_SPACE_ID,
    api_name: str | None = DEFAULT_API_NAME,
    extra_guidance: str | None = None,
    hf_token: str | None = None,
) -> str:
    """Décrit une photo d'intérieur en prompt texte pour DiT360, via Qwen-VL.

    Args:
        source: chemin, image PIL, ou ``IngestedImage``.
        client: client gradio (injectable pour les tests). Par défaut, en
            construit un sur ``space_id``.
        space_id: Space Hugging Face hébergeant le VLM.
        api_name: endpoint gradio nommé (fallback ``fn_index=0`` sinon).
        extra_guidance: instructions supplémentaires (style, contraintes…)
            ajoutées au prompt.
        hf_token: token Hugging Face optionnel (sinon ``HF_TOKEN``) pour
            relever le quota ZeroGPU du Space.

    Returns:
        Le prompt descriptif (chaîne non vide).
    """
    prompt = DESCRIBE_PROMPT
    if extra_guidance:
        prompt = f"{prompt}\n\n{extra_guidance}"

    image_path = _prepare_image_file(source)
    try:
        engine = (
            client
            if client is not None
            else _default_client(space_id, hf_token=hf_token)
        )
        file_arg = _as_file_arg(image_path)

        # Tente l'endpoint nommé puis retombe sur fn_index=0 (cf. avertissement).
        try:
            result = engine.predict(file_arg, prompt, api_name=api_name)
        except Exception:
            try:
                result = engine.predict(file_arg, prompt, fn_index=0)
            except Exception as exc:  # noqa: BLE001 - dépend du réseau/Space
                raise CaptionError(
                    f"Appel du Space {space_id} échoué : {exc}"
                ) from exc
    finally:
        image_path.unlink(missing_ok=True)

    return _to_text(result)
