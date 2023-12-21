import numpy as np
import logging
import cv2
import sys

from PIL import Image
from typing import List, Tuple


mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from . import processing_array as pa
else:
    # Code executed as a script
    import processing_array as pa
    
def is_readable_image(image_path:str)->None:
    """
    Check if an image is readable using OpenCV.

    Parameters:
    - image_path (str): The path to the image file.

    Raises:
    - ValueError: If the image is empty or cannot be read. Check if the path is correct.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"The image is empty. Please check if the path {image_path} is correct ")

def get_size_of_image(image_path):
    img_ref = Image.open(image_path)
    return img_ref.size
   
def save_image_from_array(pixels,nom_fichier_sortie: str, shape:Tuple[int,int]=[]) -> None:
    
    if shape==[]:
        if len(np.shape(pixels))!=3:
            raise ValueError(f"Incorrect shape got {np.shape(pixels)} and expected (x,y,z)")
        if np.shape(pixels)[2]!=3:
            raise ValueError(f"Incorrect dimension for the array, expected 3 and given {np.shape(pixels)[2]}")
        image = Image.fromarray(pixels.astype(np.uint8))
    else:
        if np.shape(shape)!=(2,):
            raise ValueError(f"Incorrect shape {shape} for the display")
        pa.line_to_3Darray(pixels,(shape[0],shape[1]))
        image = Image.new("RGB", (shape[0],shape[1]))

        # Remplissage de l'image avec les pixels de la liste
        # Convertit les listes en tuples
        pixel_data = [tuple((int(pixel[0]),int(pixel[1]),int(pixel[2]))) for pixel in pixels]
        image.putdata(pixel_data)
    
    # Sauvegarde de l'image
    image.save(nom_fichier_sortie)
    logging.info(f"Image saved under the name '{nom_fichier_sortie}'.")

def give_array_from_image(image_name: str):
    """
    Convertit une image en une liste de pixels (composantes RVB).

    Parameters:
    - image_name (str): Chemin vers le fichier image.

    Returns:
    - liste_pixels (list): Liste de pixels (composantes RVB).
    """
    image = Image.open(image_name)
    tableau_image = np.array(image)
    
    # Obtenir les dimensions de l'image
    largeur, hauteur, _ = tableau_image.shape

    # Reshape le tableau pour correspondre aux dimensions de l'image
    tableau_image = tableau_image.reshape((hauteur, largeur, -1))
    
    # Extraire les composantes RGB
    liste_pixels_rgb = [tuple(pixel)
                        for ligne in tableau_image for pixel in ligne]

    return np.array(liste_pixels_rgb)

def get_homography_between_imgs(image1_path: str, image2_path: str,display=False) -> np.ndarray:
    """
    Trouve la matrice d'homographie entre deux images en utilisant la correspondance de caractéristiques.

    Parameters:
    - image1_path (str): Chemin de l'image de référence.
    - image2_path (str): Chemin de l'image d'environnement.

    Returns:
    - numpy.ndarray: Matrice d'homographie.
    """
    is_readable_image(image1_path)
    is_readable_image(image2_path)
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
    if display:
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
        cv2.imshow(f'{image1_path} in {image2_path}', img2_with_reference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return H

def get_hsv_mask_with_sliders(image_path: str) -> Tuple[Tuple[int]]:
    """
    Permet à l'utilisateur de sélectionner un masque HSV à partir d'une image en utilisant des curseurs.

    Parameters:
    - image_path (str): Chemin vers l'image source.

    Returns:
    - tuple: Couple du masque HSV sélectionné (lower_hsv, upper_hsv).
    """
    is_readable_image(image_path)
    
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
        return np.array(lower_hsv), np.array(upper_hsv)

    # Charger votre image
    image = cv2.imread(image_path)

    
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
    logging.info(f"Mask hsv [{lower_hsv},{upper_hsv}] exported !")
    return np.array(lower_hsv), np.array(upper_hsv)

def get_shining_point_image(image_path: np.ndarray,display=False) -> Tuple[int]:

    is_readable_image(image_path)
    # Charger votre image
    image_array = cv2.imread(image_path)
    
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
    if display:
        # Visualisation
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Dessine un cercle rouge

        # # Afficher l'image avec le point brillant détecté
        cv2.imshow('Shining point', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x, y

def get_shining_point_with_hsv_mask(image_path: str,hsv_mask:np.ndarray, display: bool = False):
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
    is_readable_image(image_path)
    
    lower_red=hsv_mask[0]
    upper_red=hsv_mask[1]

    # Charger l'image
    image = cv2.imread(image_path)

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

    if display:
        # Dessiner les contours sur l'image originale
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Afficher l'image avec les contours
        cv2.imshow('Image avec contours du point laser', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pixel_x, pixel_y

if __name__ == '__main__':
    # print(get_shining_point_image("image_source.png"))
    # array=give_array_from_image("image_source.png")
    # save_image_from_array(array,"image_source_rebuilt.png",(640,480))
    H=get_homography_between_imgs("image_ref.png","image_source.png",True)
    print(H)