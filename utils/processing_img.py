import numpy as np
import logging

from PIL import Image
from typing import List, Tuple

def save_PIL_image(image, name_image:str):
    # Sauvegarde de l'image
    image.save(name_image)
    logging.info(f"Image saved under the name '{name_image}'.")
    
def save_PIL_image_from_array(liste_pixels: List[int], largeur: int, hauteur: int, nom_fichier_sortie: str) -> None:
    """
    Crée une image à partir d'une liste de pixels et la sauvegarde dans un fichier.

    Parameters:
    - liste_pixels (list): Liste des pixels au format RGB.
    - largeur (int): Largeur de l'image.
    - hauteur (int): Hauteur de l'image.
    - nom_fichier_sortie (str): Nom du fichier de sortie.
    """
    image = Image.new("RGB", (largeur, hauteur))

    # Remplissage de l'image avec les pixels de la liste
    # Convertit les listes en tuples
    pixel_data = [tuple(pixel) for pixel in liste_pixels]
    image.putdata(pixel_data)
    
    save_PIL_image(image,nom_fichier_sortie)

def give_array_from_PIL_image(image) -> List[Tuple[int, int, int]]:
    """
    Convertit une image en une liste de pixels (composantes RVB).

    Parameters:
    - image_name (str): Chemin vers le fichier image.

    Returns:
    - liste_pixels (list): Liste de pixels (composantes RVB).
    """

    tableau_image = np.array(image)
    
    # Obtenir les dimensions de l'image
    largeur, hauteur, _ = tableau_image.shape

    # Reshape le tableau pour correspondre aux dimensions de l'image
    tableau_image = tableau_image.reshape((hauteur, largeur, -1))
    
    # Extraire les composantes RGB
    liste_pixels_rgb = [tuple(pixel)
                        for ligne in tableau_image for pixel in ligne]

    return liste_pixels_rgb

def give_array_from_image_name(image_name: str) -> List[Tuple[int, int, int]]:
    """
    Convertit une image en une liste de pixels (composantes RVB).

    Parameters:
    - image_name (str): Chemin vers le fichier image.

    Returns:
    - liste_pixels (list): Liste de pixels (composantes RVB).
    """
    image = Image.open(image_name)
    return give_array_from_PIL_image(image)


if __name__ == '__main__':
    array=give_array_from_image_name("test.png")
    save_PIL_image_from_array(array,640,480,"test2.png")