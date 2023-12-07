
import numpy as np

try:
    import realsense.acquisition as aq
except:
    import acquisition as aq
try:
    import realsense.utils.homography as h
except:
    import utils.homography as h

from PIL import Image

from typing import Tuple


def pc_centered_on_centre_objet_ref(image1_path: str, image2_path: str, longueur: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repose le nuage de points acquis par la caméra Realsense par rapport au centre de l'objet de référence.

    Parameters:
    - image1_path (str): Nom de l'image de référence à charger.
    - image2_path (str): Nom pour sauvegarder l'image d'acquisition de la caméra (aussi nommée image source).
    - longueur (int): Largeur de l'image (en pixels) prise par la Realsense.

    Returns:
    - Tuple: Un tuple contenant le nuage de points centré autour du centre de l'objet de référence et l'image couleur acquise par la caméra Realsense.
    """

    img_ref = Image.open(image1_path)
    largeur, hauteur = img_ref.size

    centre_image_ref = np.array(
        [int(largeur/2), int(hauteur/2), 1])  # En pixel

    # On fait l'acquisition avec la caméra
    vertices, color_image = aq.points_and_colors_realsense(image2_path)

    # On calcule les coordonnées (toujours en pixel) de ce centre dans l'image source
    # Pour cela on calcule la matrice d'homographie
    homography_matrix = h.feature_matching(image1_path, image2_path)

    projected_center = np.dot(homography_matrix, centre_image_ref)  # en pixel

    # On repasse en coordonnées en profondeur

    origine_nuage = vertices[(int(projected_center[1])-1)
                             * longueur + int(projected_center[0])]

    # On repose ensuite le nuage de point pour que ce dernier soit centré autour du point origine_nuage
    # (qui correpont au centre de l'objet de l'image de reference)

    nuage_point_centred = [(x[0] - origine_nuage[0], x[1] -
                            origine_nuage[1], x[2] - origine_nuage[2]) for x in vertices]

    # cv.create_ply_file(nuage_point_centred,
    #                    rpc.colors_relasense_sofa(color_image), "test.ply")
    return nuage_point_centred, color_image
