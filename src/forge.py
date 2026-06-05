"""Point d'entrée du pipeline pano-forge : photos → monde 3D World Labs.

Objectif : un monde 3D **fidèle à la pièce photographiée**. La fidélité vient du
conditionnement par image — World Labs Marble génère le monde directement à
partir d'une (ou plusieurs) image(s) réelle(s).

Deux backends :

- ``worldlabs`` (défaut) — envoie la/les **vraie(s) photo(s)** à World Labs en
  ``image_prompt`` (fidèle aux pixels). Avec ``--multi``, jusqu'à 4 meilleures
  photos en ``multi-image`` (upload media-asset, comme image-blaster).
- ``dit360`` — ancien chemin : caption (Qwen-VL) → panorama DiT360 (texte→pano,
  générique) → World Labs. Conservé en option.

Flux : ``ingest → [sélection photos | caption+DiT360] → World Labs → assets``.

CLI :

    python -m src.forge input/ --submit                 # photo réelle -> World Labs
    python -m src.forge input/ --multi --submit         # 4 photos -> multi-image
    python -m src.forge input/ --backend dit360 --submit  # ancien chemin DiT360
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from src.caption import CaptionError, describe_room
from src.ingest import MIN_IMAGES, IngestError, ingest
from src.panorama import (
    DEFAULT_SEED,
    DEFAULT_STEPS,
    Panorama,
    PanoramaError,
    generate_panorama,
)

# API World Labs (Marble). Auth via header WLT-Api-Key ; clé dans l'environnement.
WORLD_LABS_ENDPOINT = "https://api.worldlabs.ai/marble/v1"
WORLD_LABS_MODEL = "marble-1.1"
WORLD_LABS_API_KEY_ENV = "WORLD_LABS_API_KEY"

# marble-1.1 multi-image accepte au plus 4 images de prompt.
MULTI_IMAGE_LIMIT = 4

# Backends de pipeline.
BACKEND_WORLDLABS = "worldlabs"
BACKEND_DIT360 = "dit360"
DEFAULT_BACKEND = BACKEND_WORLDLABS

# Bord long max des photos envoyées (orientation EXIF déjà corrigée par ingest).
MAX_SOURCE_EDGE = 2048

# Mode --mix : durée d'une photo fixe, nombre de keyframes vidéo gardées, et
# durée d'affichage d'un keyframe (court montage des angles les plus distincts).
DEFAULT_STILL_SECONDS = 5.0
DEFAULT_MIX_FRAMES = 50
MIX_KEYFRAME_SECONDS = 0.2

# --mix qualité : seuil de netteté (variance du Laplacien) sous lequel une image
# est jugée trop floue et écartée ; et luminance moyenne sous laquelle une photo
# est jugée « sombre » et ré-égalisée pour coller aux frames vidéo.
DEFAULT_MIN_SHARPNESS = 50.0
DARK_LUMA_THRESHOLD = 100.0

# --mix : seuil de mouvement (différence inter-frame) sous lequel une frame est
# jugée immobile (caméra arrêtée) et écartée ; nombre de frames de fondu enchaîné
# entre clips ; côté court minimum d'une photo avant upscale LANCZOS4.
DEFAULT_MIN_MOTION = 5.0
DEFAULT_TRANSITION_FRAMES = 8
MIN_PHOTO_SHORT_SIDE = 720

# Codecs d'encodage du MP4 mix, du plus compact (H.264) au plus compatible. On
# garde le premier dont le VideoWriter s'ouvre réellement.
MIX_CODECS = ("avc1", "H264", "mp4v", "XVID")

# Validation vidéo (--mix) : formats acceptés, résolution et durée admissibles.
SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".mkv")
MIN_VIDEO_SHORT_SIDE = 480
MIN_VIDEO_SECONDS = 3.0
MAX_VIDEO_SECONDS = 300.0

# Métriques thumbnail : variance de Laplacien par bloc sous laquelle un bloc est
# jugé flou (pour le ratio de zones floues).
THUMBNAIL_BLUR_BLOCK_THRESHOLD = 100.0

# Retry World Labs : un 500/503 transitoire est retenté quelques fois avant abandon.
WORLD_LABS_RETRY_STATUSES = (500, 503)
WORLD_LABS_MAX_RETRIES = 3
WORLD_LABS_RETRY_DELAY = 10.0


class WorldLabsError(Exception):
    """Erreur levée lors d'un échec d'appel à l'API World Labs."""


class VideoValidationError(Exception):
    """Vidéo refusée par la validation (format, résolution ou durée invalide)."""


@dataclass
class PipelineResult:
    """Résultat complet d'un passage du pipeline."""

    backend: str
    source_images: list[Path]
    run_dir: Path | None = None  # sous-dossier dédié du run (output/<timestamp>/)
    prompt: str | None = None
    panorama: Panorama | None = None
    world_labs_request: dict = field(default_factory=dict)
    world: dict | None = None  # réponse World Labs si --submit
    world_assets: dict | None = None  # assets téléchargés si --submit


# ---------------------------------------------------------------------------
# Sélection / export des photos sources
# ---------------------------------------------------------------------------


def _color_histogram(pil_image: Image.Image, *, bins: int = 8) -> np.ndarray:
    """Histogramme couleur RGB normalisé — descripteur perceptuel léger.

    L'image est réduite (rapidité) ; on concatène les histogrammes des 3 canaux
    et on normalise. Deux vues très similaires (même mur, même cadrage) ont des
    histogrammes proches ; des angles différents divergent.
    """
    img = pil_image.convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float64)
    channels = [
        np.histogram(arr[:, :, c], bins=bins, range=(0, 255))[0] for c in range(3)
    ]
    hist = np.concatenate(channels)
    total = hist.sum()
    return hist / total if total else hist


def _diverse_indices(feats: list[np.ndarray], k: int) -> list[int]:
    """Farthest-point sampling : indices des ``k`` descripteurs les plus diversifiés.

    Distance L1 entre histogrammes. Amorçage déterministe sur le descripteur le
    plus éloigné de la moyenne (le plus distinctif), puis ajout glouton de
    l'image qui maximise la distance minimale à l'ensemble déjà choisi.
    """
    matrix = np.asarray(feats)
    n = len(matrix)

    def l1(i: int, j: int) -> float:
        return float(np.abs(matrix[i] - matrix[j]).sum())

    mean = matrix.mean(axis=0)
    seed = int(np.argmax(np.abs(matrix - mean).sum(axis=1)))
    selected = [seed]
    while len(selected) < k:
        best_idx, best_dist = None, -1.0
        for j in range(n):
            if j in selected:
                continue
            d = min(l1(j, s) for s in selected)
            if d > best_dist:
                best_dist, best_idx = d, j
        selected.append(best_idx)
    return selected


def _select_images(images: list, *, multi: bool) -> list:
    """Sélectionne les photos sources pour World Labs.

    - **single** : la plus haute résolution (meilleure qualité d'ancrage).
    - **multi** : les ``MULTI_IMAGE_LIMIT`` photos les plus **diversifiées** en
      vue/angle (histogrammes couleur + farthest-point sampling), pour couvrir
      au mieux la pièce plutôt que 4 quasi-doublons.
    """
    if not multi:
        return sorted(
            images, key=lambda im: (-(im.width * im.height), im.path.name.lower())
        )[:1]
    if len(images) <= MULTI_IMAGE_LIMIT:
        return list(images)
    feats = [_color_histogram(im.image) for im in images]
    return [images[i] for i in _diverse_indices(feats, MULTI_IMAGE_LIMIT)]


def _export_image(pil_image: Image.Image, dest: str | Path, *, max_edge: int = MAX_SOURCE_EDGE) -> Path:
    """Écrit une image (réduite si besoin) en JPEG vers ``dest``."""
    img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
    longest = max(img.width, img.height)
    if longest > max_edge:
        scale = max_edge / longest
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=92)
    return dest


def _even_azimuth(index: int, count: int) -> int:
    return round(index * 360 / count) if count else 0


def _upscale_if_low_res(
    image_bgr: np.ndarray, *, min_side: int = MIN_PHOTO_SHORT_SIDE
) -> np.ndarray:
    """Upscale (cv2 INTER_LANCZOS4) une image dont le côté court est sous ``min_side``.

    Les photos basse résolution sont agrandies avant insertion pour coller à la
    résolution vidéo (LANCZOS4 = interpolation de meilleure qualité d'OpenCV).
    """
    import cv2

    h, w = image_bgr.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return image_bgr
    scale = min_side / short
    return cv2.resize(
        image_bgr,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )


def _letterbox_bgr(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Insère une image BGR dans un canvas WxH (letterbox, bandes noires).

    Agrandit (LANCZOS4) ou réduit (INTER_AREA) selon le facteur d'échelle.
    """
    import cv2

    h0, w0 = image_bgr.shape[:2]
    scale = min(width / w0, height / h0)
    nw, nh = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=interp)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def _sharpness(image_bgr: np.ndarray) -> float:
    """Score de netteté = variance du Laplacien (faible => image floue)."""
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_luma(image_bgr: np.ndarray) -> float:
    """Luminance moyenne (0-255) d'une image BGR."""
    import cv2

    return float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean())


def _auto_brightness_bgr(
    image_bgr: np.ndarray, *, dark_threshold: float = DARK_LUMA_THRESHOLD
) -> np.ndarray:
    """Égalise l'histogramme de luminance des images sombres.

    Seules les images dont la luminance moyenne est sous ``dark_threshold`` sont
    corrigées (egalisation du canal Y en YCrCb, les couleurs sont préservées),
    pour que les photos sombres collent à la luminosité des frames vidéo.
    """
    import cv2

    if _mean_luma(image_bgr) >= dark_threshold:
        return image_bgr
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _select_motion_frames(
    scores: list[float], max_frames: int, *, eligible: list[int] | None = None
) -> list[int]:
    """Indices des frames au plus fort changement visuel, en ordre temporel.

    ``scores[i]`` = différence inter-frame de la frame ``i`` (mouvement caméra).
    On garde les ``max_frames`` plus mouvementées — les **vrais nouveaux angles**
    — puis on les remet en ordre chronologique pour un montage cohérent.
    ``eligible`` restreint la sélection à un sous-ensemble (frames assez nettes).
    """
    candidates = list(range(len(scores))) if eligible is None else list(eligible)
    if not candidates:
        return []
    keep = min(max_frames, len(candidates))
    top = sorted(candidates, key=lambda i: (scores[i], i), reverse=True)[:keep]
    return sorted(top)


def _assign_photos_to_keyframes(
    photo_feats: list[np.ndarray], key_feats: list[np.ndarray]
) -> dict[int, list[int]]:
    """Place chaque photo après le keyframe le plus similaire (distance L1).

    Retourne ``{position_keyframe: [indices_photos]}`` : chaque photo est insérée
    au moment le plus pertinent de la timeline plutôt qu'à la fin.
    """
    assignments: dict[int, list[int]] = {}
    if not key_feats:
        return assignments
    for p_idx, pf in enumerate(photo_feats):
        dists = [float(np.abs(pf - kf).sum()) for kf in key_feats]
        best = int(np.argmin(dists))
        assignments.setdefault(best, []).append(p_idx)
    return assignments


def _motion_blur_bgr(image: np.ndarray, *, kernel_size: int | None = None) -> np.ndarray:
    """Léger flou de mouvement horizontal : rapproche une photo fixe d'une frame.

    World Labs intègre mieux des stills qui ressemblent à des frames vidéo
    naturelles (légèrement filées) qu'à des images parfaitement nettes. Noyau
    linéaire horizontal, taille proportionnelle au petit côté (donc « léger »).
    """
    import cv2

    h, w = image.shape[:2]
    if kernel_size is None:
        kernel_size = max(3, round(min(w, h) / 120))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


# ---------------------------------------------------------------------------
# Vidéo : validation, orientation, extraction (avec cache) et encodage
# ---------------------------------------------------------------------------


def _video_rotation(cap) -> int:
    """Rotation (0/90/180/270°) déclarée dans les métadonnées de la vidéo.

    Lit ``CAP_PROP_ORIENTATION_META`` (rotation track / EXIF vidéo). Renvoie 0 si
    indisponible ou non standard.
    """
    import cv2

    try:
        rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META)))
    except (AttributeError, ValueError, TypeError):  # pragma: no cover
        return 0
    rot %= 360
    return rot if rot in (90, 180, 270) else 0


def _rotate_bgr(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Applique une rotation de 0/90/180/270° à une frame BGR."""
    import cv2

    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def validate_video(path: str | Path) -> dict:
    """Valide une vidéo avant traitement ; lève ``VideoValidationError`` sinon.

    Contrôles : extension (mp4/mov/mkv), lisibilité, résolution (côté court
    ≥ 480p, après prise en compte de la rotation) et durée (entre 3 s et 300 s).
    Retourne les métadonnées de base (fps, frame_count, dimensions, durée).
    """
    import cv2

    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTS:
        raise VideoValidationError(
            f"Format vidéo non supporté : {ext or '(aucune extension)'} "
            f"(attendu : {', '.join(SUPPORTED_VIDEO_EXTS)})."
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(f"Vidéo illisible ou codec non supporté : {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotation = _video_rotation(cap)
    cap.release()

    if rotation in (90, 270):
        width, height = height, width
    if min(width, height) < MIN_VIDEO_SHORT_SIDE:
        raise VideoValidationError(
            f"Résolution trop basse ({width}x{height}) ; "
            f"minimum {MIN_VIDEO_SHORT_SIDE}p."
        )
    if fps <= 0 or count <= 0:
        raise VideoValidationError(
            "Durée indéterminable (fps ou nombre de frames manquant)."
        )
    duration = count / fps
    if not (MIN_VIDEO_SECONDS <= duration <= MAX_VIDEO_SECONDS):
        raise VideoValidationError(
            f"Durée {duration:.1f}s hors bornes "
            f"[{MIN_VIDEO_SECONDS:.0f}, {MAX_VIDEO_SECONDS:.0f}]s."
        )
    return {
        "fps": fps,
        "frame_count": int(count),
        "width": width,
        "height": height,
        "duration": duration,
        "rotation": rotation,
    }


def _video_file_hash(path: str | Path) -> str:
    """Hash MD5 du fichier vidéo (clé de cache des frames extraites)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_path(frames_dir: Path, index: int) -> Path:
    return frames_dir / f"frame_{index:06d}.jpg"


def _extract_frames(video_path: str | Path, frames_dir: str | Path) -> dict:
    """Extrait les frames (rotation corrigée) vers ``frames_dir``, avec cache.

    Si ``frames_dir`` contient déjà un ``meta.json`` cohérent, les frames sont
    réutilisées telles quelles (cache hit). Sinon la vidéo est décodée : chaque
    frame est redressée selon la rotation, sauvegardée en JPEG, et les scores de
    mouvement (différence inter-frame) et de netteté (Laplacien) sont calculés.
    """
    import cv2

    frames_dir = Path(frames_dir)
    meta_path = frames_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = None
        if meta and meta.get("count") and _frame_path(frames_dir, meta["count"] - 1).exists():
            return meta  # cache hit : frames déjà extraites pour ce hash

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    # Désactive l'auto-rotation d'OpenCV pour appliquer la rotation nous-mêmes.
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except (AttributeError, cv2.error):  # pragma: no cover
        pass
    rotation = _video_rotation(cap)

    scores: list[float] = []
    sharp: list[float] = []
    prev_gray = None
    count = 0
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _rotate_bgr(frame, rotation)
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        scores.append(0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean()))
        sharp.append(_sharpness(frame))
        prev_gray = gray
        cv2.imwrite(str(_frame_path(frames_dir, count)), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    cap.release()
    if count == 0:
        raise WorldLabsError(f"Vidéo sans frames lisibles : {video_path}")

    meta = {
        "fps": fps,
        "width": width,
        "height": height,
        "count": count,
        "rotation": rotation,
        "scores": scores,
        "sharpness": sharp,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _open_video_writer(dest: Path, fps: float, width: int, height: int):
    """Ouvre un ``VideoWriter`` avec le premier codec disponible (H.264 d'abord).

    Retourne ``(writer, codec_name)``. Lève ``WorldLabsError`` si aucun codec ne
    s'initialise.
    """
    import cv2

    for name in MIX_CODECS:
        writer = cv2.VideoWriter(
            str(dest), cv2.VideoWriter_fourcc(*name), fps, (width, height)
        )
        if writer.isOpened():
            return writer, name
        writer.release()
    raise WorldLabsError(
        "Impossible d'initialiser l'encodeur MP4 (aucun codec disponible)."
    )


def _prepare_still_bgr(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Prépare une photo pour le MP4 : auto-luminosité, upscale, letterbox, flou."""
    corrected = _auto_brightness_bgr(image_bgr)
    upscaled = _upscale_if_low_res(corrected)
    boxed = _letterbox_bgr(upscaled, width, height)
    return _motion_blur_bgr(boxed)


def _image_quality_metrics(path: str | Path) -> dict:
    """Métriques de qualité d'une image : netteté, luminosité, contraste, flou.

    - **sharpness** : variance globale du Laplacien.
    - **brightness** : luminance moyenne (0-255).
    - **contrast** : écart-type des niveaux de gris.
    - **blur_ratio** : fraction de blocs (grille 8x8) dont la variance de
      Laplacien est sous ``THUMBNAIL_BLUR_BLOCK_THRESHOLD``.
    """
    import cv2

    arr = np.asarray(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    bh, bw = max(1, h // 8), max(1, w // 8)
    total = blurry = 0
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            block = lap[y : y + bh, x : x + bw]
            if block.size == 0:
                continue
            total += 1
            if float(block.var()) < THUMBNAIL_BLUR_BLOCK_THRESHOLD:
                blurry += 1
    return {
        "sharpness": float(lap.var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur_ratio": (blurry / total) if total else 0.0,
    }


def _build_mix_video(
    video_path: str | Path,
    photos: list[Image.Image],
    dest: str | Path,
    *,
    still_seconds: float = DEFAULT_STILL_SECONDS,
    max_frames: int = DEFAULT_MIX_FRAMES,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    min_motion: float = DEFAULT_MIN_MOTION,
    transition_frames: int = DEFAULT_TRANSITION_FRAMES,
    cache_dir: str | Path | None = None,
    stats: dict | None = None,
) -> Path:
    """Assemble un MP4 (sans audio) : keyframes vidéo mouvementés + photos.

    1. **Extraction + cache** : les frames sont décodées (rotation corrigée) et
       mises en cache par hash MD5 sous ``cache_dir`` ; une vidéo resoumise est
       relue depuis le cache.
    2. **Zones immobiles** : les frames dont le mouvement est sous ``min_motion``
       (caméra arrêtée) sont écartées.
    3. **Mouvement + netteté** : parmi les frames mouvementées et nettes
       (Laplacien ≥ ``min_sharpness``), on garde les ``max_frames`` au plus fort
       changement visuel (garde-fou : repli sur toutes les frames si vide).
    4. **Photos** : photos floues écartées, sombres ré-égalisées, basse
       résolution upscalées (LANCZOS4), letterbox, léger flou de mouvement.
    5. **Transitions** : ``transition_frames`` frames de fondu enchaîné autour de
       chaque photo. Encodage H.264 (repli mp4v/XVID), multi-threadé.

    Imports paresseux d'``cv2`` / ``tqdm`` (requis seulement pour ``--mix``).
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dépend de l'install
        raise WorldLabsError(
            "opencv-python est requis pour --mix. "
            "Installe-le : pip install opencv-python-headless"
        ) from exc
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm optionnel
        tqdm = None

    cv2.setNumThreads(max(1, os.cpu_count() or 1))  # encodage/filtrage multi-threadé

    # Cache persistant (hash MD5) si cache_dir fourni, sinon dossier temporaire.
    tmp_dir = None
    if cache_dir is not None:
        frames_dir = Path(cache_dir) / _video_file_hash(video_path)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="panoforge-frames-")
        frames_dir = Path(tmp_dir)

    try:
        meta = _extract_frames(video_path, frames_dir)
        fps = meta["fps"]
        width, height, count = meta["width"], meta["height"], meta["count"]
        scores, sharp = meta["scores"], meta["sharpness"]

        # Écarte frames immobiles (mouvement faible) et floues ; garde-fou si vide.
        eligible = [
            i
            for i in range(count)
            if scores[i] >= min_motion and sharp[i] >= min_sharpness
        ]
        if not eligible:
            eligible = list(range(count))
        selected = _select_motion_frames(scores, max_frames, eligible=eligible)
        keyframes = [cv2.imread(str(_frame_path(frames_dir, i))) for i in selected]

        # Photos : écarte les floues, intercale par similarité, prépare les stills.
        photo_bgr = [
            cv2.cvtColor(np.asarray(p.convert("RGB")), cv2.COLOR_RGB2BGR) for p in photos
        ]
        kept = [i for i, bgr in enumerate(photo_bgr) if _sharpness(bgr) >= min_sharpness]
        key_feats = [
            _color_histogram(Image.fromarray(cv2.cvtColor(kf, cv2.COLOR_BGR2RGB)))
            for kf in keyframes
        ]
        photo_feats = [_color_histogram(photos[i]) for i in kept]
        photo_after = _assign_photos_to_keyframes(photo_feats, key_feats)
        stills = [_prepare_still_bgr(photo_bgr[i], width, height) for i in kept]

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        writer, codec = _open_video_writer(dest, fps, width, height)

        key_hold = max(1, round(MIX_KEYFRAME_SECONDS * fps))
        still_hold = max(1, round(still_seconds * fps))

        # Clips ordonnés : (image, durée, est_une_photo).
        clips: list[tuple[np.ndarray, int, bool]] = []
        for pos, keyframe in enumerate(keyframes):
            clips.append((keyframe, key_hold, False))
            for p_idx in photo_after.get(pos, []):
                clips.append((stills[p_idx], still_hold, True))

        # Fondu uniquement aux frontières impliquant une photo (montage net sinon).
        def _needs_fade(i: int) -> bool:
            return transition_frames > 0 and (clips[i][2] or clips[i - 1][2])

        total = sum(hold for _, hold, _ in clips)
        total += sum(transition_frames for i in range(1, len(clips)) if _needs_fade(i))

        pbar = (
            tqdm(total=total, desc="Encodage MP4 mix", unit="frame")
            if tqdm is not None
            else None
        )
        try:
            for i, (img, hold, _is_photo) in enumerate(clips):
                if i > 0 and _needs_fade(i):
                    prev_img = clips[i - 1][0]
                    for step in range(1, transition_frames + 1):
                        alpha = step / (transition_frames + 1)
                        writer.write(
                            cv2.addWeighted(prev_img, 1.0 - alpha, img, alpha, 0.0)
                        )
                    if pbar is not None:
                        pbar.update(transition_frames)
                for _ in range(hold):
                    writer.write(img)
                if pbar is not None:
                    pbar.update(hold)
        finally:
            writer.release()
            if pbar is not None:
                pbar.close()

        if stats is not None:
            sel_sharp = [sharp[i] for i in selected]
            stats.update(
                {
                    "frames_extracted": count,
                    "frames_retained": len(keyframes),
                    "photos_total": len(photos),
                    "photos_kept": len(kept),
                    "mean_sharpness": float(np.mean(sel_sharp)) if sel_sharp else 0.0,
                    "codec": codec,
                    "audio": False,
                    "transition_frames": transition_frames,
                    "min_motion": min_motion,
                    "fps": fps,
                    "resolution": [width, height],
                    "rotation": meta["rotation"],
                    "mp4_bytes": dest.stat().st_size if dest.exists() else 0,
                    "cached": cache_dir is not None,
                }
            )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return dest


# ---------------------------------------------------------------------------
# Appels HTTP World Labs
# ---------------------------------------------------------------------------


def _request_json(url: str, *, api_key: str, method: str = "GET", payload: dict | None = None) -> dict:
    """Appel JSON minimal vers l'API World Labs (header WLT-Api-Key).

    Les erreurs serveur transitoires (``WORLD_LABS_RETRY_STATUSES``, p. ex. 500 /
    503) sont retentées jusqu'à ``WORLD_LABS_MAX_RETRIES`` fois, avec
    ``WORLD_LABS_RETRY_DELAY`` secondes d'attente entre chaque tentative.
    """
    data = None
    headers = {"WLT-Api-Key": api_key}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if exc.code in WORLD_LABS_RETRY_STATUSES and attempt < WORLD_LABS_MAX_RETRIES:
                attempt += 1
                time.sleep(WORLD_LABS_RETRY_DELAY)
                continue
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise WorldLabsError(
                f"World Labs {method} a échoué ({exc.code}) : {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise WorldLabsError(f"World Labs injoignable : {exc}") from exc
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError:
        return {}


def _put_file(url: str, path: str | Path, *, method: str = "PUT", headers: dict | None = None) -> None:
    """Upload binaire d'un fichier vers une URL signée (media-asset)."""
    raw = Path(path).read_bytes()
    req = urllib.request.Request(url, data=raw, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise WorldLabsError(f"Upload media-asset échoué : {url} ({exc})") from exc


# ---------------------------------------------------------------------------
# Construction de la requête worlds:generate
# ---------------------------------------------------------------------------


def _image_prompt_base64(path: str | Path) -> dict:
    """Contenu image inline (data_base64) pour une requête single-image."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WorldLabsError(f"Image introuvable : {path} ({exc})") from exc
    extension = path.suffix.lstrip(".").lower() or "jpg"
    mime = "image/jpeg" if extension in ("jpg", "jpeg") else f"image/{extension}"
    return {
        "source": "data_base64",
        "data_base64": base64.standard_b64encode(raw).decode(),
        "extension": extension,
        "mime_type": mime,
    }


def _upload_media_asset(
    path: str | Path, *, api_key: str, transport, upload, kind: str = "image"
) -> str:
    """Prépare et upload un media-asset (image ou vidéo), retourne son id."""
    path = Path(path)
    fallback_ext = "mp4" if kind == "video" else "png"
    extension = path.suffix.lstrip(".").lower() or fallback_ext
    prepare = transport(
        f"{WORLD_LABS_ENDPOINT}/media-assets:prepare_upload",
        api_key=api_key,
        method="POST",
        payload={"file_name": path.name, "kind": kind, "extension": extension},
    )
    asset = prepare.get("media_asset") or {}
    asset_id = (
        asset.get("media_asset_id")
        or asset.get("id")
        or prepare.get("media_asset_id")
    )
    upload_info = prepare.get("upload_info") or prepare.get("upload") or {}
    upload_url = upload_info.get("upload_url") or upload_info.get("url")
    upload_method = (upload_info.get("upload_method") or upload_info.get("method") or "PUT").upper()
    headers = upload_info.get("required_headers") or upload_info.get("headers") or {}
    if not asset_id or not upload_url:
        raise WorldLabsError(
            "Réponse prepare_upload sans media_asset_id ou upload_url."
        )
    upload(upload_url, path, method=upload_method, headers=headers)
    return asset_id


def _normalize_images(images) -> list[tuple[Path, int]]:
    """Normalise l'entrée en liste de ``(path, azimuth)``."""
    if isinstance(images, (str, Path)):
        return [(Path(images), 0)]
    items: list[tuple[Path, int]] = []
    seq = list(images)
    for i, entry in enumerate(seq):
        if isinstance(entry, (tuple, list)):
            path, azimuth = entry[0], int(entry[1])
        else:
            path, azimuth = entry, _even_azimuth(i, len(seq))
        items.append((Path(path), azimuth))
    return items


def _build_world_request(
    items: list[tuple[Path, int]],
    *,
    prompt: str | None,
    display_name: str,
    multi: bool,
    api_key: str,
    transport,
    upload,
) -> dict:
    """Construit le corps ``worlds:generate`` (single data_base64 ou multi-image)."""
    if multi or len(items) > 1:
        if len(items) > MULTI_IMAGE_LIMIT:
            raise WorldLabsError(
                f"Multi-image accepte au plus {MULTI_IMAGE_LIMIT} images "
                f"(reçu {len(items)})."
            )
        multi_prompt = []
        for path, azimuth in items:
            asset_id = _upload_media_asset(
                path, api_key=api_key, transport=transport, upload=upload
            )
            multi_prompt.append(
                {
                    "azimuth": azimuth,
                    "content": {"source": "media_asset", "media_asset_id": asset_id},
                }
            )
        world_prompt = {"type": "multi-image", "multi_image_prompt": multi_prompt}
    else:
        world_prompt = {"type": "image", "image_prompt": _image_prompt_base64(items[0][0])}

    if prompt:
        world_prompt["text_prompt"] = prompt

    return {
        "display_name": display_name,
        "model": WORLD_LABS_MODEL,
        "world_prompt": world_prompt,
    }


def _build_video_request(
    video_path: str | Path,
    *,
    prompt: str | None,
    display_name: str,
    api_key: str,
    transport,
    upload,
) -> dict:
    """Construit le corps ``worlds:generate`` pour une entrée vidéo.

    La vidéo est uploadée en media-asset (``kind: video``), référencée par
    ``video_prompt.media_asset_id`` (format confirmé par la doc World Labs).
    """
    asset_id = _upload_media_asset(
        video_path, api_key=api_key, transport=transport, upload=upload, kind="video"
    )
    world_prompt = {
        "type": "video",
        "video_prompt": {"source": "media_asset", "media_asset_id": asset_id},
    }
    if prompt:
        world_prompt["text_prompt"] = prompt
    return {
        "display_name": display_name,
        "model": WORLD_LABS_MODEL,
        "world_prompt": world_prompt,
    }


def _operation_id(operation: dict) -> str:
    op_id = (
        operation.get("operation_id")
        or operation.get("id")
        or operation.get("name")
    )
    if not op_id:
        raise WorldLabsError("Réponse World Labs sans operation_id.")
    return str(op_id).split("/")[-1]


def submit_world_labs(
    images=None,
    *,
    video: str | Path | None = None,
    prompt: str | None = None,
    display_name: str = "pano-forge",
    api_key: str | None = None,
    multi: bool = False,
    poll_interval: float = 15.0,
    max_polls: int = 240,
    _transport=None,
    _upload=None,
) -> dict:
    """Génère un monde 3D World Labs : submit puis polling jusqu'à ``done``.

    Entrée (exclusive) :
    - ``images`` : un chemin (single-image, ``data_base64``), ou une liste de
      chemins / ``(chemin, azimuth)`` (multi-image, upload media-asset) ;
    - ``video`` : un chemin de vidéo (upload media-asset ``kind: video``,
      ``world_prompt.type = "video"``).

    Logique portée d'``image-blaster`` (``generate-world.mjs``). La clé est lue
    depuis ``api_key`` ou ``WORLD_LABS_API_KEY``. Retourne ``operation.response``.
    """
    if images is None and video is None:
        raise WorldLabsError("Fournis 'images' ou 'video'.")

    api_key = api_key or os.environ.get(WORLD_LABS_API_KEY_ENV)
    if not api_key:
        raise WorldLabsError(
            f"Clé World Labs manquante : définis {WORLD_LABS_API_KEY_ENV}."
        )

    transport = _transport or _request_json
    upload = _upload or _put_file

    if video is not None:
        request = _build_video_request(
            video,
            prompt=prompt,
            display_name=display_name,
            api_key=api_key,
            transport=transport,
            upload=upload,
        )
    else:
        request = _build_world_request(
            _normalize_images(images),
            prompt=prompt,
            display_name=display_name,
            multi=multi,
            api_key=api_key,
            transport=transport,
            upload=upload,
        )

    operation = transport(
        f"{WORLD_LABS_ENDPOINT}/worlds:generate",
        api_key=api_key,
        method="POST",
        payload=request,
    )
    op_id = _operation_id(operation)

    polls = 0
    while not operation.get("done"):
        if polls >= max_polls:
            raise WorldLabsError(
                f"Délai dépassé en attendant World Labs (operation {op_id})."
            )
        time.sleep(poll_interval)
        operation = transport(
            f"{WORLD_LABS_ENDPOINT}/operations/{op_id}", api_key=api_key
        )
        polls += 1

    if operation.get("error"):
        raise WorldLabsError(
            f"Génération World Labs échouée : {operation['error']}"
        )
    response = operation.get("response")
    if not response:
        raise WorldLabsError("Opération World Labs terminée sans réponse.")
    return response


# ---------------------------------------------------------------------------
# Téléchargement des assets
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: str | Path) -> Path:
    """Télécharge ``url`` vers ``dest`` (stdlib urllib)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, OSError) as exc:
        raise WorldLabsError(f"Téléchargement échoué : {url} ({exc})") from exc
    return dest


def _ext_from_url(url: str, fallback: str) -> str:
    """Extension de fichier déduite d'une URL (sinon ``fallback``)."""
    return Path(urlparse(url).path).suffix or fallback


def download_world_assets(
    world: dict, output_dir: str | Path, *, prefix: str = "world"
) -> dict:
    """Télécharge les assets d'un monde World Labs dans ``output_dir``.

    Mêmes assets que ``image-blaster`` (``downloadWorldAssets``) : mesh GLB,
    pano équirectangulaire, thumbnail, et splats ``.spz`` (toutes résolutions).
    Retourne un dict des chemins locaux ({"glb", "pano", "thumbnail", "spz": {...}}).
    """
    assets = world.get("assets", {}) or {}
    out = Path(output_dir)
    result: dict = {"spz": {}}

    glb_url = (assets.get("mesh") or {}).get("collider_mesh_url")
    if glb_url:
        result["glb"] = _download_file(glb_url, out / f"{prefix}.glb")

    pano_url = (assets.get("imagery") or {}).get("pano_url")
    if pano_url:
        ext = _ext_from_url(pano_url, ".png")
        result["pano"] = _download_file(pano_url, out / f"{prefix}-pano{ext}")

    thumb_url = assets.get("thumbnail_url")
    if thumb_url:
        ext = _ext_from_url(thumb_url, ".webp")
        result["thumbnail"] = _download_file(thumb_url, out / f"{prefix}-thumbnail{ext}")

    spz_urls = (assets.get("splats") or {}).get("spz_urls") or {}
    for key, url in spz_urls.items():
        if not url:
            continue
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(key))
        result["spz"][key] = _download_file(url, out / f"{prefix}-{safe}.spz")

    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_id() -> str:
    """Identifiant de run horodaté (un sous-dossier propre par génération)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _world_status(world: dict | None, submit: bool) -> dict:
    """Résumé du résultat World Labs pour le log structuré."""
    if not submit:
        return {"status": "not_submitted"}
    if not world:
        return {"status": "no_response"}
    return {"status": "done", "assets": sorted((world.get("assets") or {}).keys())}


def _write_run_log(out: Path, run_log: dict) -> None:
    """Écrit le log structuré JSON du run (métriques + résultat)."""
    (out / "run.log.json").write_text(
        json.dumps(run_log, indent=2), encoding="utf-8"
    )


def _record_thumbnail_metrics(world_assets: dict | None, run_log: dict) -> None:
    """Calcule et logge les métriques qualité du thumbnail téléchargé, si présent.

    Les erreurs de calcul (image illisible, etc.) sont avalées : elles ne doivent
    jamais faire échouer un run par ailleurs réussi.
    """
    thumb = (world_assets or {}).get("thumbnail")
    if not thumb or not Path(thumb).exists():
        return
    t0 = time.perf_counter()
    try:
        run_log["thumbnail"] = _image_quality_metrics(thumb)
    except Exception as exc:  # pragma: no cover - dépend de l'asset distant
        run_log["thumbnail"] = {"error": str(exc)}
    run_log.setdefault("steps", {})["thumbnail_metrics"] = round(
        time.perf_counter() - t0, 4
    )


def forge(
    input_dir: str | Path,
    output_dir: str | Path = "output",
    *,
    backend: str = DEFAULT_BACKEND,
    multi: bool = False,
    video: str | Path | None = None,
    mix: bool = False,
    still_seconds: float = DEFAULT_STILL_SECONDS,
    frames: int = DEFAULT_MIX_FRAMES,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    min_motion: float = DEFAULT_MIN_MOTION,
    transition_frames: int = DEFAULT_TRANSITION_FRAMES,
    seed: int = DEFAULT_SEED,
    num_inference_steps: int = DEFAULT_STEPS,
    min_images: int = MIN_IMAGES,
    submit: bool = False,
    client=None,
    extra_guidance: str | None = None,
    hf_token: str | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    """Exécute le pipeline depuis un dossier de photos (ou une vidéo).

    Chaque run écrit dans un **sous-dossier dédié** ``output_dir/<run_id>/``
    (horodaté par défaut), pour garder les générations isolées et comparables.

    Args:
        input_dir: dossier des photos sources (ignoré si ``video``).
        output_dir: dossier parent ; le run écrit dans ``output_dir/<run_id>/``.
        backend: ``"worldlabs"`` (défaut, photo réelle → World Labs) ou
            ``"dit360"`` (caption → panorama DiT360 → World Labs).
        multi: en backend worldlabs, envoie jusqu'à 4 photos en multi-image.
        video: chemin d'une vidéo → World Labs en ``type: video`` (prioritaire
            sur les photos).
        seed / num_inference_steps: paramètres DiT360 (backend dit360).
        min_images: nombre minimal de photos requis.
        submit: si vrai, génère réellement le monde World Labs (paie) et
            télécharge les assets.
        client: client gradio injectable (caption, backend dit360).
        extra_guidance: consignes de style pour la caption (backend dit360).
        hf_token: token HF optionnel pour la caption (backend dit360).
        run_id: nom du sous-dossier de run (par défaut : horodatage).
        mix: assemble un MP4 combiné (les ``frames`` keyframes les plus
            mouvementés de ``video`` + chaque photo de ``input_dir`` floutée et
            intercalée ~``still_seconds`` s) et l'envoie à World Labs en mode
            vidéo (aucune limite de 4 images).
        still_seconds: durée d'affichage de chaque photo dans le MP4 mix.
        frames: nombre de keyframes vidéo gardés pour le MP4 mix.
        min_sharpness: seuil de netteté (Laplacien) sous lequel photos et frames
            sont écartées du MP4 mix.
        min_motion: seuil de mouvement (différence inter-frame) sous lequel une
            frame est jugée immobile et écartée du MP4 mix.
        transition_frames: nombre de frames de fondu enchaîné autour des photos.
    """
    out = Path(output_dir) / (run_id or _run_id())
    out.mkdir(parents=True, exist_ok=True)

    # --- mode mix : MP4 combiné (vidéo + photos en frames fixes) -> World Labs vidéo ---
    if mix:
        if video is None:
            raise ValueError("--mix nécessite --video.")
        video_path = Path(video)
        if not video_path.exists():
            raise WorldLabsError(f"Vidéo introuvable : {video_path}")

        run_log: dict = {"run_id": out.name, "backend": "mix", "video": str(video_path), "steps": {}}

        t0 = time.perf_counter()
        validate_video(video_path)
        run_log["steps"]["validate"] = round(time.perf_counter() - t0, 4)

        try:
            photos = list(ingest(input_dir, min_images=1))
        except IngestError:
            photos = []  # mix tolère l'absence de photos (vidéo seule)
        photo_images = [im.image for im in photos]

        mix_stats: dict = {}
        t0 = time.perf_counter()
        mixed_path = _build_mix_video(
            video_path,
            photo_images,
            out / "mixed.mp4",
            still_seconds=still_seconds,
            max_frames=frames,
            min_sharpness=min_sharpness,
            min_motion=min_motion,
            transition_frames=transition_frames,
            cache_dir=Path(output_dir) / ".cache",
            stats=mix_stats,
        )
        run_log["steps"]["build_mix"] = round(time.perf_counter() - t0, 4)
        run_log["mix"] = mix_stats

        manifest = {
            "endpoint": WORLD_LABS_ENDPOINT,
            "model": WORLD_LABS_MODEL,
            "backend": "mix",
            "video": str(video_path),
            "photos": len(photo_images),
            "still_seconds": still_seconds,
            "frames": frames,
            "min_sharpness": min_sharpness,
            "min_motion": min_motion,
            "transition_frames": transition_frames,
            "mixed_video": str(mixed_path),
            "prompt": None,
        }
        (out / "world_labs_request.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        world = None
        world_assets = None
        if submit:
            t0 = time.perf_counter()
            world = submit_world_labs(
                video=mixed_path, display_name=out.name or "pano-forge"
            )
            run_log["steps"]["submit"] = round(time.perf_counter() - t0, 4)
            t0 = time.perf_counter()
            world_assets = download_world_assets(world, out)
            run_log["steps"]["download"] = round(time.perf_counter() - t0, 4)
        run_log["world"] = _world_status(world, submit)
        _record_thumbnail_metrics(world_assets, run_log)
        _write_run_log(out, run_log)
        return PipelineResult(
            backend="mix",
            source_images=[mixed_path],
            run_dir=out,
            prompt=None,
            panorama=None,
            world_labs_request=manifest,
            world=world,
            world_assets=world_assets,
        )

    # --- entrée vidéo : court-circuite ingest/photos ---
    if video is not None:
        video_path = Path(video)
        if not video_path.exists():
            raise WorldLabsError(f"Vidéo introuvable : {video_path}")
        run_log = {"run_id": out.name, "backend": "video", "video": str(video_path), "steps": {}}
        manifest = {
            "endpoint": WORLD_LABS_ENDPOINT,
            "model": WORLD_LABS_MODEL,
            "backend": "video",
            "video": str(video_path),
            "prompt": None,
        }
        (out / "world_labs_request.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        world = None
        world_assets = None
        if submit:
            t0 = time.perf_counter()
            world = submit_world_labs(
                video=video_path, display_name=out.name or "pano-forge"
            )
            run_log["steps"]["submit"] = round(time.perf_counter() - t0, 4)
            t0 = time.perf_counter()
            world_assets = download_world_assets(world, out)
            run_log["steps"]["download"] = round(time.perf_counter() - t0, 4)
        run_log["world"] = _world_status(world, submit)
        _record_thumbnail_metrics(world_assets, run_log)
        _write_run_log(out, run_log)
        return PipelineResult(
            backend="video",
            source_images=[video_path],
            run_dir=out,
            prompt=None,
            panorama=None,
            world_labs_request=manifest,
            world=world,
            world_assets=world_assets,
        )

    if backend not in (BACKEND_WORLDLABS, BACKEND_DIT360):
        raise ValueError(
            f"Backend inconnu : {backend!r}. "
            f"Choisis {BACKEND_WORLDLABS!r} ou {BACKEND_DIT360!r}."
        )

    run_log = {"run_id": out.name, "backend": backend, "steps": {}}
    images = ingest(input_dir, min_images=min_images)

    prompt: str | None = None
    panorama: Panorama | None = None

    if backend == BACKEND_DIT360:
        source = _select_images(images, multi=False)[0]
        prompt = describe_room(
            source, client=client, extra_guidance=extra_guidance, hf_token=hf_token
        )
        panorama = generate_panorama(
            prompt, seed=seed, num_inference_steps=num_inference_steps
        )
        source_images = [panorama.save(out / "panorama.jpg")]
        items = [(source_images[0], 0)]
        use_multi = False
    else:  # worldlabs : photo(s) réelle(s)
        selected = _select_images(images, multi=multi)
        source_images = [
            _export_image(im.image, out / f"source-{i:02d}.jpg")
            for i, im in enumerate(selected)
        ]
        n = len(source_images)
        items = [(p, _even_azimuth(i, n)) for i, p in enumerate(source_images)]
        use_multi = multi and n > 1

    manifest = {
        "endpoint": WORLD_LABS_ENDPOINT,
        "model": WORLD_LABS_MODEL,
        "backend": backend,
        "multi": use_multi,
        "source_images": [str(p) for p in source_images],
        "prompt": prompt,
    }
    (out / "world_labs_request.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    run_log["source_images"] = len(source_images)

    world = None
    world_assets = None
    if submit:
        submit_arg = items if use_multi else items[0][0]
        t0 = time.perf_counter()
        world = submit_world_labs(
            submit_arg,
            prompt=prompt,
            display_name=out.name or "pano-forge",
            multi=use_multi,
        )
        run_log["steps"]["submit"] = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        world_assets = download_world_assets(world, out)
        run_log["steps"]["download"] = round(time.perf_counter() - t0, 4)
    run_log["world"] = _world_status(world, submit)
    _record_thumbnail_metrics(world_assets, run_log)
    _write_run_log(out, run_log)

    return PipelineResult(
        backend=backend,
        source_images=source_images,
        run_dir=out,
        prompt=prompt,
        panorama=panorama,
        world_labs_request=manifest,
        world=world,
        world_assets=world_assets,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Pipeline pano-forge : photos d'interieur -> monde 3D World Labs.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="input",
        help="Dossier des photos sources (par défaut : input/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        default="output",
        help="Dossier de sortie (par défaut : output/).",
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=[BACKEND_WORLDLABS, BACKEND_DIT360],
        default=DEFAULT_BACKEND,
        help=f"Backend (par défaut : {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help=f"Backend worldlabs : envoie jusqu'à {MULTI_IMAGE_LIMIT} photos en multi-image.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Chemin d'une vidéo à envoyer à World Labs (type video, prioritaire sur les photos).",
    )
    parser.add_argument(
        "--mix",
        action="store_true",
        help="Assemble un MP4 (--video + photos de input/ en frames fixes) et l'envoie à World Labs en vidéo.",
    )
    parser.add_argument(
        "--still-seconds",
        dest="still_seconds",
        type=float,
        default=DEFAULT_STILL_SECONDS,
        help=f"Durée d'affichage de chaque photo dans le MP4 mix (par défaut : {DEFAULT_STILL_SECONDS} s).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_MIX_FRAMES,
        help=f"Nombre de keyframes vidéo gardés pour le MP4 mix (par défaut : {DEFAULT_MIX_FRAMES}).",
    )
    parser.add_argument(
        "--min-sharpness",
        dest="min_sharpness",
        type=float,
        default=DEFAULT_MIN_SHARPNESS,
        help=f"Seuil de netteté (Laplacien) sous lequel photos/frames sont écartées du mix (par défaut : {DEFAULT_MIN_SHARPNESS}).",
    )
    parser.add_argument(
        "--min-motion",
        dest="min_motion",
        type=float,
        default=DEFAULT_MIN_MOTION,
        help=f"Seuil de mouvement sous lequel une frame immobile est écartée du mix (par défaut : {DEFAULT_MIN_MOTION}).",
    )
    parser.add_argument(
        "--transition-frames",
        dest="transition_frames",
        type=int,
        default=DEFAULT_TRANSITION_FRAMES,
        help=f"Frames de fondu enchaîné autour des photos dans le MP4 mix (par défaut : {DEFAULT_TRANSITION_FRAMES}).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Graine DiT360 (par défaut : {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--steps",
        dest="num_inference_steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Étapes de diffusion DiT360 (par défaut : {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "-m",
        "--min-images",
        type=int,
        default=MIN_IMAGES,
        help=f"Nombre minimal de photos requis (par défaut : {MIN_IMAGES}).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=f"Génère le monde World Labs (nécessite {WORLD_LABS_API_KEY_ENV}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Le prompt/les chemins peuvent contenir des caractères hors cp1252 ; force
    # UTF-8 sur les consoles Windows pour éviter un UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    args = _build_parser().parse_args(argv)
    try:
        result = forge(
            args.input_dir,
            args.output_dir,
            backend=args.backend,
            multi=args.multi,
            video=args.video,
            mix=args.mix,
            still_seconds=args.still_seconds,
            frames=args.frames,
            min_sharpness=args.min_sharpness,
            min_motion=args.min_motion,
            transition_frames=args.transition_frames,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            min_images=args.min_images,
            submit=args.submit,
        )
    except (
        IngestError,
        CaptionError,
        PanoramaError,
        WorldLabsError,
        VideoValidationError,
        ValueError,
    ) as exc:
        print(f"[forge] Erreur : {exc}")
        return 1

    print(f"[forge] Backend : {result.backend}")
    print(f"[forge] Dossier du run : {result.run_dir}")
    for path in result.source_images:
        print(f"[forge] Source : {path}")
    if result.prompt:
        print(f"[forge] Prompt : {result.prompt}")

    if result.world is not None:
        assets = result.world.get("assets", {})
        print(f"[forge] Monde World Labs généré (assets : {list(assets)}).")
        downloaded = result.world_assets or {}
        for kind in ("glb", "pano", "thumbnail"):
            if downloaded.get(kind):
                print(f"[forge]   {kind}: {downloaded[kind]}")
        for key, path in (downloaded.get("spz") or {}).items():
            print(f"[forge]   spz[{key}]: {path}")
    else:
        print(
            f"[forge] Requête World Labs préparée "
            f"({args.output_dir}/world_labs_request.json). "
            f"Ajoute --submit (+ {WORLD_LABS_API_KEY_ENV}) pour générer le monde."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
