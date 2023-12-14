import numpy as np
import open3d as o3d
import logging

from typing import List, Tuple, Optional

import processing_files as pf
import processing_array as pa

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

def plot_ply(path_name: str) -> None:
    """
    Plot 3D points and triangles from PLY file vertices.

    Parameters:
    - path_name (string): The name of the ply to plot

    Returns:
    - None
    """
    points,colors = get_points_and_colors_of_ply(path_name)
    vertices = [point + color for point, color in zip(points, colors)]
    pa.plot_ply_from_array(np.array(vertices))
    
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

def save_ply_from_map(map_file: str, ply_file: str) -> None:
    """
    Convertit un fichier .map en un fichier .ply contenant un nuage de points.

    Parameters:
    - map_file (str): Chemin du fichier .map d'entrée.
    - ply_file (str): Chemin du fichier .ply de sortie.
    """
    lines = pf.open_file_and_give_content(map_file)

    # Extraire les coordonnées XYZ
    points = [list(map(float, line.strip().split())) for line in lines]

    # Convertir la liste en tableau NumPy
    points_np = np.array(points, dtype=np.float32)

    save_ply_file(ply_file,points_np)

def color_ply_depending_on_axis(name_ply:str,new_name:str,axis:str):
    points,_=get_points_and_colors_of_ply(name_ply)
    colors=pa.color_3D_array_depending_on_axis(points,axis)
    save_ply_file(new_name,points,colors)

if __name__ == '__main__':
    color_ply_depending_on_axis("test.ply","test_colore.ply","z")
    # points,colors=get_points_and_colors_of_ply('test.ply')
    # save_ply_file("with_colors.ply",points,colors)
    # save_ply_file("without_colors.ply",points)
    # save_ply_from_map("test.map","ply_from_map.ply")