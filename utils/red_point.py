import cv2
import numpy as np

from typing import Tuple


def laser_in_picture(image_array: np.ndarray) -> Tuple[int]:
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


def detect_point_rouge_hsv(path_image: str, lower_red: np.ndarray, upper_red: np.ndarray, Affichage: bool = False):
    """
    Retourne les coordonnées en pixel d'un point rouge présent dans une image à partir d'un masque HSV.

    Parameters:
    - path_image (str): Chemin vers l'image à analyser.
    - lower_red (numpy.ndarray): Plage inférieure de la couleur rouge dans l'espace HSV.
    - upper_red (numpy.ndarray): Plage supérieure de la couleur rouge dans l'espace HSV.
    - Affichage (bool): Indique si l'image avec les contours du point laser doit être affichée.

    Returns:
    - tuple: Coordonnées (pixel_x, pixel_y) du point rouge détecté.
    """
    # Charger l'image
    image = cv2.imread(path_image)

    # Convertir l'image en espace de couleur HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Créer un masque en utilisant la plage de couleurs définie
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Trouver les contours dans le masque
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_sum = 0
    y_sum = 0
    sum = 0
    # Parcourir tous les contours détectés
    for contour in contours:
        # Récupérer les coordonnées des pixels dans le contour
        for point in contour:
            x, y = point[0]
            x_sum += x
            y_sum += y
            sum += 1
    pixel_x = int(x_sum/sum)
    pixel_y = int(y_sum/sum)

    if Affichage:
        # Dessiner les contours sur l'image originale
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Afficher l'image avec les contours
        cv2.imshow('Image avec contours du point laser', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pixel_x, pixel_y
