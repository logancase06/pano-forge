"""Étape 3 du pipeline : forge du monde World Labs.

Prend un panorama 360° équirectangulaire et produit l'asset final attendu par
World Labs (monde 3D navigable).

À implémenter dans une prochaine itération.
"""

from __future__ import annotations


def forge(panorama, output_dir: str = "output"):
    """Construit le monde 3D à partir d'un panorama 360°.

    Args:
        panorama: Image panoramique équirectangulaire produite par :mod:`src.panorama`.
        output_dir: Dossier de sortie pour l'asset World Labs.
    """
    raise NotImplementedError("La forge World Labs n'est pas encore implémentée.")
