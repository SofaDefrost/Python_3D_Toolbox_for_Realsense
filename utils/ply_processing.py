import numpy as np
import open3d as o3d
import logging

from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from typing import List, Tuple, Optional

import file_manager as fm

def get_points_and_colors_of_ply(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convertit un fichier .ply en nuage de points et en couleur au format RGB entre 0 et 255.

    Parameters:
    - file_path (str): Chemin du fichier .ply.

    Returns:
    - tuple: Un tuple contenant un tableau numpy des points et un tableau numpy des couleurs.
    """
    # Charge les données du fichier .ply
    ply_data = o3d.io.read_point_cloud(file_path)

    # Récupère les points et les couleurs
    points = np.array(ply_data.points)
    colors = np.array(ply_data.colors) * 255

    return points, colors

def plot_ply_from_array(vertices: np.ndarray) -> None:
    """
    Plot 3D points and triangles from PLY file vertices.

    Parameters:
    - vertices (np.ndarray): An array containing vertices with X, Y, Z coordinates.
      If the array has 6 columns, it is assumed to include color information (RGB).

    Returns:
    - None
    """
    # Extract color information if available
    if vertices.shape[1] == 6:
        # Normalize colors to the range [0, 1]
        colors = vertices[:, 3:6] / 255.0
    else:
        colors = 'b'  # Use blue as default color if no color information is present

    # Perform Delaunay triangulation
    triangles = Delaunay(vertices[:, :2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices with colors
    ax.scatter(vertices[:, 0], vertices[:, 1],
               vertices[:, 2], c=colors, marker='o')

    # Plot the triangles
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=triangles.simplices, color='gray', alpha=0.5)

    plt.show()

def plot_ply_from_path(path_name: str) -> None:
    """
    Plot 3D points and triangles from PLY file vertices.

    Parameters:
    - path_name (string): The name of the ply to plot

    Returns:
    - None
    """
    points,colors = get_points_and_colors_of_ply(path_name)
    vertices = [point + color for point, color in zip(points, colors)]
    plot_ply_from_array(np.array(vertices))
    
def save_ply_file(output_filename: str, points: np.ndarray, colors: Optional[np.ndarray] = []) -> None:
    """
    Crée un fichier PLY à partir de points et de couleurs, avec un en-tête spécifique.

    Parameters:
    - output_filename (str): Nom du fichier PLY de sortie.
    - points (numpy.ndarray): Tableau des points.
    - colors (numpy.ndarray): Tableau des couleurs au format RGB.

    Raises:
    - ValueError: Si le nombre de points ne correspond pas au nombre de couleurs.
    """
    with_color=False
    if len(colors)>0:
        with_color=True
        
    if (len(points) != len(colors)) and  with_color:
        raise ValueError(
            "Le nombre de points doit correspondre au nombre de couleurs.")

    with open(output_filename, 'w') as ply_file:
        # Écriture de l'en-tête PLY
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")

        if with_color:
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")

        ply_file.write("end_header\n")
        
        # Écriture des données de points et couleurs
        if with_color:
            for point, color in zip(points, colors):
                x = point[0]
                y = point[1]
                z = point[2]
                r, g, b = color
                ply_file.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        else:
            for point in points:
                x = point[0]
                y = point[1]
                z = point[2]
                ply_file.write(f"{x} {y} {z}\n")

        logging.info(f"Le fichier '{output_filename}' a été créé.")

def creer_image_a_partir_de_liste(liste_pixels: List[int], largeur: int, hauteur: int, nom_fichier_sortie: str) -> None:
    """
    Crée une image à partir d'une liste de pixels et la sauvegarde dans un fichier.

    Parameters:
    - liste_pixels (list): Liste des pixels au format RGB.
    - largeur (int): Largeur de l'image.
    - hauteur (int): Hauteur de l'image.
    - nom_fichier_sortie (str): Nom du fichier de sortie.
    """
    # Création de l'image à partir de la liste de pixels
    image = Image.new("RGB", (largeur, hauteur))

    # Remplissage de l'image avec les pixels de la liste
    # Convertit les listes en tuples
    pixel_data = [tuple(pixel) for pixel in liste_pixels]
    image.putdata(pixel_data)

    # Sauvegarde de l'image
    image.save(nom_fichier_sortie)

    print(f"L'image a été créée et sauvegardée sous '{nom_fichier_sortie}'.")


def image_en_liste(chemin_image: str) -> List[Tuple[int, int, int]]:
    """
    Convertit une image en une liste de pixels (composantes RVB).

    Parameters:
    - chemin_image (str): Chemin vers le fichier image.

    Returns:
    - liste_pixels (list): Liste de pixels (composantes RVB).
    """
    # Ouvrir l'image avec Pillow
    image = Image.open(chemin_image)

    # Convertir l'image en tableau NumPy
    tableau_image = np.array(image)

    # Obtenir les dimensions de l'image
    largeur, hauteur, _ = tableau_image.shape

    # Reshape le tableau pour correspondre aux dimensions de l'image
    tableau_image = tableau_image.reshape((hauteur, largeur, -1))

    # Extraire les composantes RVB
    liste_pixels_rgb = [tuple(pixel[:-1])
                        for ligne in tableau_image for pixel in ligne]

    return liste_pixels_rgb

def convert_map_to_ply(map_file: str, ply_file: str) -> None:
    """
    Convertit un fichier .map en un fichier .ply contenant un nuage de points.

    Parameters:
    - map_file (str): Chemin du fichier .map d'entrée.
    - ply_file (str): Chemin du fichier .ply de sortie.
    """
    # Lire le fichier .map
    with open(map_file, 'r') as f:
        lines = f.readlines()

    # Extraire les coordonnées XYZ
    points = [list(map(float, line.strip().split())) for line in lines]

    # Convertir la liste en tableau NumPy
    points_np = np.array(points, dtype=np.float32)

    # Créer un nuage de points Open3D
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np[:, :3])

    # Sauvegarder le nuage de points au format .ply
    o3d.io.write_point_cloud(ply_file, point_cloud)

    print(f"Le nuage de points a été sauvegardé sous '{ply_file}'.")

if __name__ == '__main__':
    points,colors=get_points_and_colors_of_ply('test.ply')
    save_ply_file("with_colors.ply",points,colors)
    save_ply_file("without_colors.ply",points)