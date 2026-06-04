"""Pont ingest → panorama : décrit une photo source en prompt texte riche.

Le Space DiT360 est texte→panorama : il génère un équirectangulaire à partir
d'un prompt. Pour partir d'une *photo* d'intérieur, on décrit d'abord la pièce
avec un modèle de vision (Claude, via l'API Anthropic), puis ce prompt alimente
:class:`src.panorama.HFSpaceBackend`.

Usage :

    from src.caption import describe_room
    from src.panorama import generate_panorama

    prompt = describe_room("input/salon.jpg")
    pano = generate_panorama(prompt)

L'import de l'SDK ``anthropic`` est paresseux : ce module s'importe sans la
dépendance, et la clé n'est lue qu'au moment de l'appel
(``ANTHROPIC_API_KEY`` dans l'environnement).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

# Modèle de vision par défaut (capable de vision, le plus capable disponible).
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 1024

# Extensions → media_type pour l'envoi base64 à l'API.
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Consigne stable, mise en cache (préfixe partagé entre toutes les photos).
# Le prompt doit décrire la pièce comme une scène 360° pour un modèle
# texte→panorama équirectangulaire (DiT360).
SYSTEM_PROMPT = (
    "Tu es un assistant qui rédige des prompts pour un modèle de génération "
    "de panoramas 360° équirectangulaires (DiT360). On te montre une photo "
    "normale d'un intérieur. Décris la PIÈCE ENTIÈRE comme une scène à 360°, "
    "pas seulement le cadrage de la photo : agencement et disposition des murs, "
    "mobilier et sa position, sols et plafonds, matériaux et textures, palette "
    "de couleurs, sources et ambiance lumineuse, style de décoration, et "
    "atmosphère générale. Extrapole de façon plausible les zones hors-champ "
    "pour former un espace cohérent et continu sur 360°.\n\n"
    "Réponds UNIQUEMENT par le prompt descriptif, en une seule phrase ou un "
    "court paragraphe dense, en anglais, sans préambule ni guillemets."
)

USER_INSTRUCTION = (
    "Décris cette pièce comme un prompt de panorama 360° équirectangulaire."
)


class CaptionError(Exception):
    """Erreur levée lorsque la génération du prompt échoue."""


def _encode_image(source: str | Path | Image.Image | object) -> tuple[str, str]:
    """Normalise une source en ``(media_type, données_base64)``.

    Accepte un chemin, une image PIL, ou un objet exposant ``.image`` (par ex.
    :class:`src.ingest.IngestedImage`).
    """
    # IngestedImage (ou tout objet portant une image PIL) → on prend son image.
    if not isinstance(source, (str, Path, Image.Image)) and hasattr(source, "image"):
        source = source.image

    if isinstance(source, Image.Image):
        img = source if source.mode == "RGB" else source.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return "image/jpeg", base64.standard_b64encode(buffer.getvalue()).decode()

    path = Path(source)
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise CaptionError(
            f"Type d'image non supporté pour {path.name!r}. "
            f"Extensions : {', '.join(sorted(_MEDIA_TYPES))}"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CaptionError(f"Impossible de lire l'image : {path} ({exc})") from exc
    return media_type, base64.standard_b64encode(data).decode()


def _default_client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dépend de l'install
        raise CaptionError(
            "Le paquet 'anthropic' est requis pour la caption. "
            "Installe-le : pip install anthropic"
        ) from exc
    # Résout ANTHROPIC_API_KEY depuis l'environnement.
    return anthropic.Anthropic()


def describe_room(
    source: str | Path | Image.Image | object,
    *,
    client=None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extra_guidance: str | None = None,
) -> str:
    """Décrit une photo d'intérieur en prompt texte pour DiT360.

    Args:
        source: chemin, image PIL, ou ``IngestedImage``.
        client: client Anthropic (injectable pour les tests). Par défaut, en
            construit un qui lit ``ANTHROPIC_API_KEY``.
        model: modèle de vision à utiliser.
        max_tokens: plafond de tokens pour la description.
        extra_guidance: instructions supplémentaires (style, contraintes…)
            ajoutées à la consigne utilisateur.

    Returns:
        Le prompt descriptif (chaîne non vide).
    """
    media_type, image_b64 = _encode_image(source)
    engine = client if client is not None else _default_client()

    instruction = USER_INSTRUCTION
    if extra_guidance:
        instruction = f"{instruction}\n\n{extra_guidance}"

    try:
        response = engine.messages.create(
            model=model,
            max_tokens=max_tokens,
            # Préfixe stable mis en cache : amorti sur de nombreuses photos.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - on enveloppe toute erreur SDK/réseau
        raise CaptionError(f"Appel à l'API Anthropic échoué : {exc}") from exc

    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    if not text:
        raise CaptionError("La réponse du modèle ne contient pas de texte.")
    return text
