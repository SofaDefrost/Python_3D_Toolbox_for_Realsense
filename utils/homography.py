import cv2
import numpy as np

# Retourne la matrice d'hommogarphie entre deux images
def feature_matching(image1_path, image2_path):
    # Charger les images
    img1 = cv2.imread(image1_path, 0)  # Image de référence en niveaux de gris
    img2 = cv2.imread(image2_path, 0)  # Image d'environnement en niveaux de gris

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
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Trouver la matrice d'homographie
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H

