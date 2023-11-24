import cv2
import numpy as np


def laser_in_picture(image_array):
    """
    Détecte le point laser le plus brillant dans une image.

    Parameters:
    - image_array (numpy.ndarray): Tableau représentant une image en format RGB.

    Returns:
    - tuple: Coordonnées (x, y) du point laser le plus brillant détecté.
    """
    # Convertir le tableau en une image OpenCV
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Utiliser la fonction cornerHarris pour détecter le coin le plus brillant
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Normaliser la réponse pour faciliter la détection du point le plus brillant
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)

    # Trouver les coordonnées du point brillant
    y, x = np.unravel_index(dst.argmax(), dst.shape)

    # Visualisation
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Dessine un cercle rouge

    # # Afficher l'image avec le point brillant détecté
    # cv2.imshow('Point Brillant', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return x, y
