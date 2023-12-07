import cv2
import numpy as np


def feature_matching(image1_path: str, image2_path: str) -> np.ndarray:
    """
    Trouve la matrice d'homographie entre deux images en utilisant la correspondance de caractéristiques.

    Parameters:
    - image1_path (str): Chemin de l'image de référence.
    - image2_path (str): Chemin de l'image d'environnement.

    Returns:
    - numpy.ndarray: Matrice d'homographie.
    """
    # Charger les images
    img1 = cv2.imread(image1_path, 0)  # Image de référence en niveaux de gris
    # Image d'environnement en niveaux de gris
    img2 = cv2.imread(image2_path, 0)

    # Initialiser l'ORB detector
    orb = cv2.ORB_create()

    # Trouver les points clés et les descripteurs avec ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Utiliser le BFMatcher pour trouver les meilleures correspondances
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Trier les correspondances en fonction de leur similarité
    matches = sorted(matches, key=lambda x: x.distance)

    # Sélectionner les meilleures correspondances (peut ajuster le ratio en conséquence)
    good_matches = matches[:50]

    # Obtenir les points correspondants dans les deux images
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Trouver la matrice d'homographie
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Charger les images en couleur
    img1_color = cv2.imread(image1_path)
    img2_color = cv2.imread(image2_path)

    # Appliquer la matrice d'homographie pour transformer les coins de l'image de référence
    h, w = img1_color.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(pts, H)

    # Dessiner les contours de l'image de référence dans l'image source
    img2_with_reference = cv2.polylines(img2_color, [np.int32(transformed_pts)], True, (0, 255, 0), 2)

    # Afficher l'image résultante
    cv2.imshow('Image avec image de référence', img2_with_reference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return H

# Exemple d'utilisation
image1_path = 'images/image_ref.png'
image2_path = 'images/image.png'
homography_matrix = feature_matching(image1_path, image2_path)

