"""Étape 2 du pipeline : assemblage panorama.

Prend les images ingérées (voir :mod:`src.ingest`) et les assemble en une
projection panoramique équirectangulaire 360°.

À implémenter dans une prochaine itération.
"""

from __future__ import annotations

from src.ingest import IngestedImage


def stitch(images: list[IngestedImage]):
    """Assemble une liste d'images en un panorama équirectangulaire.

    Returns:
        L'image panoramique assemblée.
    """
    raise NotImplementedError("L'assemblage panorama n'est pas encore implémenté.")
