"""Étape 2 du pipeline : génération du panorama 360°.

Le moteur de génération est **DiT360** (Insta360 Research Team, CVPR 2026,
modèle ``Insta360-Research/DiT360-Panorama-Image-Generation``).

DiT360 n'est PAS exposé sur l'Inference API serverless de Hugging Face. Deux
chemins existent, abstraits ici derrière :class:`PanoramaBackend` :

- :class:`HFSpaceBackend` — appelle le Space Gradio hébergé (gratuit, sans GPU)
  via ``gradio_client``. Le Space est **texte→panorama** : il prend un prompt
  et rend un équirectangulaire 2048×1024.
- :class:`LocalDiT360Backend` — fait tourner DiT360 en local via ``diffusers``
  (GPU CUDA requis). Supporte texte→panorama aujourd'hui ; le conditionnement
  par image (outpainting depuis une photo source) est prévu mais pas encore
  implémenté.

Les dépendances lourdes (``gradio_client``, ``diffusers``, ``torch``) sont
importées paresseusement pour que l'import de ce module reste léger.

Usage :

    from src.panorama import generate_panorama
    pano = generate_panorama("un salon lumineux, style scandinave")
    pano.image.save("output/pano.jpg")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

# Modèle / Space DiT360 par défaut.
DEFAULT_SPACE_ID = "Insta360-Research/DiT360"
DEFAULT_MODEL_ID = "Insta360-Research/DiT360-Panorama-Image-Generation"

# Le Space fixe la sortie en équirectangulaire 2:1.
PANO_WIDTH = 2048
PANO_HEIGHT = 1024
DEFAULT_STEPS = 50
DEFAULT_SEED = 0


class PanoramaError(Exception):
    """Erreur levée lorsqu'un backend ne peut pas produire de panorama."""


@dataclass
class Panorama:
    """Un panorama 360° équirectangulaire généré."""

    image: Image.Image
    width: int
    height: int
    prompt: str
    backend: str

    @property
    def is_equirectangular(self) -> bool:
        """Vrai si le ratio est ~2:1 (tolérance 1 %)."""
        if self.height == 0:
            return False
        return abs(self.width / self.height - 2.0) < 0.02

    def save(self, path: str | Path, quality: int = 95) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(dest, quality=quality)
        return dest


class PanoramaBackend(Protocol):
    """Contrat commun à tous les moteurs de génération de panorama."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        seed: int = DEFAULT_SEED,
        num_inference_steps: int = DEFAULT_STEPS,
    ) -> Panorama:
        """Génère un panorama équirectangulaire à partir d'un prompt texte."""
        ...


def _to_pil(result) -> Image.Image:
    """Normalise la sortie d'un appel Gradio en image PIL.

    ``gradio_client`` peut renvoyer un chemin de fichier, un dict
    ({"path": ...} / {"url": ...}), un tuple, ou directement une image.
    """
    # Gradio renvoie souvent (tuple/list) plusieurs sorties ; on prend la 1re.
    if isinstance(result, (list, tuple)):
        if not result:
            raise PanoramaError("Le Space n'a renvoyé aucune sortie.")
        result = result[0]

    if isinstance(result, Image.Image):
        return result
    if isinstance(result, dict):
        result = result.get("path") or result.get("url")
        if result is None:
            raise PanoramaError("Sortie Gradio sans chemin ni URL exploitable.")
    if isinstance(result, (str, Path)):
        try:
            return Image.open(result).convert("RGB")
        except OSError as exc:
            raise PanoramaError(
                f"Image générée illisible : {result} ({exc})"
            ) from exc

    raise PanoramaError(f"Type de sortie Gradio inattendu : {type(result)!r}")


class HFSpaceBackend:
    """Backend gratuit : appelle le Space DiT360 via ``gradio_client``.

    Le Space est texte→panorama. Aucun GPU local requis ; en contrepartie,
    file d'attente et quotas du Space s'appliquent.

    ``client_factory`` permet d'injecter un faux client pour les tests.
    """

    name = "hf_space"

    def __init__(
        self,
        space_id: str = DEFAULT_SPACE_ID,
        *,
        api_name: str | None = "/infer",
        hf_token: str | None = None,
        client_factory=None,
    ) -> None:
        self.space_id = space_id
        self.api_name = api_name
        self.hf_token = hf_token
        self._client_factory = client_factory
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(self.space_id)
            return self._client
        try:
            from gradio_client import Client
        except ImportError as exc:  # pragma: no cover - dépend de l'install
            raise PanoramaError(
                "gradio_client est requis pour le backend Space. "
                "Installe-le : pip install gradio_client"
            ) from exc
        if self.hf_token:
            # Le nom du kwarg a changé selon les versions : token (>=2.x) vs
            # hf_token (1.x).
            try:
                self._client = Client(self.space_id, token=self.hf_token)
            except TypeError:
                self._client = Client(self.space_id, hf_token=self.hf_token)
        else:
            self._client = Client(self.space_id)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        seed: int = DEFAULT_SEED,
        num_inference_steps: int = DEFAULT_STEPS,
    ) -> Panorama:
        if not prompt or not prompt.strip():
            raise PanoramaError("Le backend Space exige un prompt texte non vide.")

        client = self._get_client()
        args = (prompt, seed, num_inference_steps)

        # Le Space utilise gr.Blocks sans api_name garanti : on tente l'api_name,
        # puis on retombe sur fn_index=0 si l'endpoint nommé n'existe pas.
        try:
            result = client.predict(*args, api_name=self.api_name)
        except Exception:
            try:
                result = client.predict(*args, fn_index=0)
            except Exception as exc:  # pragma: no cover - dépend du réseau/Space
                raise PanoramaError(
                    f"Appel du Space {self.space_id} échoué : {exc}"
                ) from exc

        image = _to_pil(result)
        return Panorama(
            image=image,
            width=image.width,
            height=image.height,
            prompt=prompt,
            backend=self.name,
        )


class LocalDiT360Backend:
    """Backend local : DiT360 via ``diffusers`` (GPU CUDA requis).

    Texte→panorama est implémenté. Le conditionnement par image (outpainting
    depuis une photo source) est prévu mais pas encore branché — voir
    :meth:`generate_from_image`.
    """

    name = "local_dit360"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:  # pragma: no cover - dépend de l'install
            raise PanoramaError(
                "diffusers et torch sont requis pour le backend local. "
                "Installe-les : pip install diffusers torch"
            ) from exc
        torch_dtype = getattr(torch, self.dtype, torch.float32)
        self._pipe = DiffusionPipeline.from_pretrained(
            self.model_id, dtype=torch_dtype, device_map=self.device
        )
        return self._pipe

    def generate(
        self,
        prompt: str,
        *,
        seed: int = DEFAULT_SEED,
        num_inference_steps: int = DEFAULT_STEPS,
    ) -> Panorama:
        if not prompt or not prompt.strip():
            raise PanoramaError("Le backend local exige un prompt texte non vide.")

        import torch

        pipe = self._get_pipe()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = pipe(
            prompt,
            width=PANO_WIDTH,
            height=PANO_HEIGHT,
            num_inference_steps=num_inference_steps,
            guidance_scale=2.8,
            generator=generator,
        ).images[0]
        return Panorama(
            image=image,
            width=image.width,
            height=image.height,
            prompt=prompt,
            backend=self.name,
        )

    def generate_from_image(
        self,
        source: Image.Image,
        *,
        prompt: str = "",
        seed: int = DEFAULT_SEED,
        num_inference_steps: int = DEFAULT_STEPS,
    ) -> Panorama:
        """Image→panorama par outpainting depuis une photo source.

        Pas encore implémenté : nécessite le pipeline d'outpainting de DiT360.
        """
        raise NotImplementedError(
            "Le conditionnement par image (outpainting DiT360) n'est pas encore "
            "implémenté. Utilise generate(prompt=...) pour l'instant."
        )


_BACKENDS = {
    HFSpaceBackend.name: HFSpaceBackend,
    LocalDiT360Backend.name: LocalDiT360Backend,
}


def build_backend(name: str = HFSpaceBackend.name, **kwargs) -> PanoramaBackend:
    """Construit un backend par nom : ``"hf_space"`` (défaut) ou ``"local_dit360"``."""
    try:
        cls = _BACKENDS[name]
    except KeyError:
        valid = ", ".join(sorted(_BACKENDS))
        raise PanoramaError(
            f"Backend inconnu : {name!r}. Disponibles : {valid}."
        ) from None
    return cls(**kwargs)


def generate_panorama(
    prompt: str,
    *,
    backend: str | PanoramaBackend = HFSpaceBackend.name,
    seed: int = DEFAULT_SEED,
    num_inference_steps: int = DEFAULT_STEPS,
    **backend_kwargs,
) -> Panorama:
    """Génère un panorama 360° à partir d'un prompt texte avec DiT360.

    ``backend`` accepte un nom (``"hf_space"`` / ``"local_dit360"``) ou une
    instance déjà construite de :class:`PanoramaBackend`.
    """
    engine = backend if not isinstance(backend, str) else build_backend(
        backend, **backend_kwargs
    )
    return engine.generate(
        prompt, seed=seed, num_inference_steps=num_inference_steps
    )
