import numpy as np
import acquisition as get
import filtre_hsv_realsense as filtre
from utils import hsv as apply_hsv
from utils import filtrage_bruit as bruit
from utils import convert as cv

# Fichier d'exemple qui montre comment utilser les différentes fonctions.
# A l'éxecution du programme une acquisition est effectuée, avec un filtrage hsv + filtrage anti bruit
# A la fin est exporté un fichier .ply de l'acquisition


def points_realsense_sofa(vertices):
    """
    Convertit les points de la caméra RealSense en un format lisible par Sofa.

    Parameters:
    - vertices (numpy.ndarray): Tableau des coordonnées XYZ des points de la caméra RealSense.

    Returns:
    - numpy.ndarray: Tableau des points formatés avec des colonnes supplémentaires pour qx, qy, qz, qw.
    """
    points = []
    valid_points_list = vertices.tolist()
    for i in range(len(valid_points_list)):
        l = [valid_points_list[i][0], valid_points_list[i]
             [1], valid_points_list[i][2]]
        # on crée une liste points qui va stocker les informations, [0.,0.,0.,0.] correspond aux cordonée qx qy qz qw
        points.append(l+[0., 0., 0., 0.])
    return np.array(points)


def colors_relasense_sofa(colors):
    """
    Convertit les couleurs de la caméra RealSense en un format lisible par Sofa.

    Parameters:
    - colors (numpy.ndarray): Tableau des couleurs de la caméra RealSense.

    Returns:
    - numpy.ndarray: Tableau des couleurs formaté pour Sofa.
    """
    l = len(colors)*len(colors[0])
    new_colors = np.asarray([(0, 0, 0) for i in range(l)])
    indice = 0
    for i in range(len(colors)):
        for j in range(len(colors[0])):
            new_colors[indice] = colors[i][j]
            indice += 1
    return np.array(new_colors)


# On détermine le masque à appliquer
MASK = filtre.determine_mask_hsv()
# On récupère les points et les couleurs de la caméra
VERTICES, COLORS = get.points_and_colors_realsense()
# On converit les couleurs dans un bon format (ie on met l'image en ligne)
COLORS_SOFA = colors_relasense_sofa(COLORS)
# On applique le masque
NEW_POINTS, NEW_COLORS = apply_hsv.mask(VERTICES, COLORS_SOFA, MASK)

NEW_POINTS_SOFA = points_realsense_sofa(NEW_POINTS)  # Attention
# On extrait la partie que l'on veut
NEW_POINTS_SURFACE = [sous_liste[:3] for sous_liste in NEW_POINTS_SOFA]

# On filtre le bruit
FINAL_POINTS, FINAL_COLOR = bruit.interface_de_filtrage_de_points(
    np.array(NEW_POINTS_SURFACE), NEW_POINTS)
# On enregistre on format .ply
cv.create_ply_file(FINAL_POINTS, FINAL_COLOR, "test.ply")
