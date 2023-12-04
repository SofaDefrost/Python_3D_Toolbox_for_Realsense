import numpy as np

import convert as cv

from typing import List, Optional, Tuple


def statistiques_hauteur(liste_hauteurs: List[float]) -> Optional[Tuple[float, float, float]]:
    """
    Calcule la hauteur maximale, minimale et moyenne à partir d'une liste de hauteurs.

    Args:
        liste_hauteurs (list): Liste des hauteurs à analyser.

    Returns:
        tuple or None: Un tuple contenant la hauteur minimale, moyenne et maximale, 
                      ou None si la liste est vide.
    """
    if not liste_hauteurs:
        return None  # Retourne None si la liste est vide

    hauteur_max = max(liste_hauteurs)
    hauteur_min = min(liste_hauteurs)
    hauteur_moyenne = sum(liste_hauteurs) / len(liste_hauteurs)

    return hauteur_min, hauteur_moyenne, hauteur_max


def hauteur_vers_rgb(hauteur: float, plage_basse: float, plage_moyenne: float, plage_elevee: float) -> Tuple[int, int, int]:
    """
    Convertit une hauteur en valeurs RGB en fonction des plages spécifiées.

    Args:
        hauteur (float): La hauteur à convertir.
        plage_basse (float): La limite inférieure de la plage basse.
        plage_moyenne (float): La limite supérieure de la plage basse et la limite 
                              inférieure de la plage moyenne.
        plage_elevee (float): La limite supérieure de la plage moyenne et la limite 
                              inférieure de la plage élevée.

    Returns:
        tuple: Un tuple représentant les valeurs RGB de la couleur correspondant à la hauteur.
    """
    couleur_basse = (0, 0, 255)  # Bleu pour les basses hauteurs
    couleur_moyenne = (0, 255, 0)  # Vert pour les hauteurs moyennes
    couleur_elevee = (255, 0, 0)  # Rouge pour les hautes hauteurs

    # Convertir la hauteur en valeurs RGB
    if hauteur <= plage_basse:
        return couleur_basse
    elif plage_basse < hauteur <= plage_moyenne:
        ratio = (hauteur - plage_basse) / (plage_moyenne - plage_basse)
        couleur = (
            int((1 - ratio) * couleur_basse[0] + ratio * couleur_moyenne[0]),
            int((1 - ratio) * couleur_basse[1] + ratio * couleur_moyenne[1]),
            int((1 - ratio) * couleur_basse[2] + ratio * couleur_moyenne[2])
        )
        return couleur
    elif plage_moyenne < hauteur <= plage_elevee:
        ratio = (hauteur - plage_moyenne) / (plage_elevee - plage_moyenne)
        couleur = (
            int((1 - ratio) * couleur_moyenne[0] + ratio * couleur_elevee[0]),
            int((1 - ratio) * couleur_moyenne[1] + ratio * couleur_elevee[1]),
            int((1 - ratio) * couleur_moyenne[2] + ratio * couleur_elevee[2])
        )
        return couleur
    else:
        return couleur_elevee

# Exemple d'application
# path_file="foie_V.ply"
# points,_ =cv.ply_to_points_and_colors(path_file)
# couleurs=np.array([ [0, 0, 0] for i in range(len(points))])
# coordonnees_z = [coord[2] for coord in points]
# valeur_hauteur=statistiques_hauteur(coordonnees_z)
# print(valeur_hauteur)
# for i in range(len(couleurs)):
#     couleurs[i]=hauteur_vers_rgb(points[i][2],valeur_hauteur[0],valeur_hauteur[1],valeur_hauteur[2])
# print(couleurs)
# cv.create_ply_file(points,couleurs,"foie_V_couleurs_h.ply")
