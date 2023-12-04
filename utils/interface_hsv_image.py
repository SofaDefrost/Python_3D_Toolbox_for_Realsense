import cv2
import numpy as np

from typing import Tuple


def interface_hsv_image(image_path: str) -> Tuple[Tuple[int]]:
    """
    Permet à l'utilisateur de sélectionner un masque HSV à partir d'une image en utilisant des curseurs.

    Parameters:
    - image_path (str): Chemin vers l'image source.

    Returns:
    - tuple: Couple du masque HSV sélectionné (lower_hsv, upper_hsv).
    """
    def update_mask_hsv(x):
        global lower_hsv, upper_hsv, mask, result
        lower_hsv = np.array([cv2.getTrackbarPos('H_L', 'HSV Interface'),
                              cv2.getTrackbarPos('S_L', 'HSV Interface'),
                              cv2.getTrackbarPos('V_L', 'HSV Interface')])

        upper_hsv = np.array([cv2.getTrackbarPos('H_U', 'HSV Interface'),
                              cv2.getTrackbarPos('S_U', 'HSV Interface'),
                              cv2.getTrackbarPos('V_U', 'HSV Interface')])

        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
        cv2.imshow('HSV Mask', mask)

        # Appliquer le masque sur l'image originale
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('Result', result)
        return lower_hsv, upper_hsv

    # Charger votre image
    image = cv2.imread(image_path)

    if image is None:
        print("L'image n'a pas pu être chargée. Assurez-vous que le chemin de l'image est correct.")
    else:
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Créer les fenêtres
        cv2.namedWindow('Original Image')
        cv2.namedWindow('HSV Interface')
        cv2.namedWindow('HSV Mask')
        cv2.namedWindow('Result')

        # Créer les curseurs pour les composantes HSV
        cv2.createTrackbar('H_L', 'HSV Interface', 0, 179, update_mask_hsv)
        cv2.createTrackbar('S_L', 'HSV Interface', 0, 255, update_mask_hsv)
        cv2.createTrackbar('V_L', 'HSV Interface', 0, 255, update_mask_hsv)
        cv2.createTrackbar('H_U', 'HSV Interface', 179, 179, update_mask_hsv)
        cv2.createTrackbar('S_U', 'HSV Interface', 255, 255, update_mask_hsv)
        cv2.createTrackbar('V_U', 'HSV Interface', 255, 255, update_mask_hsv)

        # Initialiser les plages HSV
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([179, 255, 255])
        # Créez une image noire comme masque initial
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        result = image.copy()

        # Afficher l'image initiale
        cv2.imshow('Original Image', image)

        while True:
            # Mettre à jour le masque en fonction des curseurs
            lower_hsv, upper_hsv = update_mask_hsv(0)

            # Afficher les trois fenêtres
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Une fois que l'utilisateur appuie sur 'q', renvoyer le masque
        cv2.destroyAllWindows()
        print('Masque exporté ! ')
        return lower_hsv, upper_hsv

# interface_hsv_image("image_source.png")
