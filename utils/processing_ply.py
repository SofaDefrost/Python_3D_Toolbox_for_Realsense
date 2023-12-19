import numpy as np
import open3d as o3d
import logging

from typing import List, Tuple, Optional

import processing_general as pg
import processing_array as pa

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
    pa.plot_3D_array(np.array(vertices))
    
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
    if len(output_filename)<5:
        raise ValueError(f"Incorrect filename {output_filename}")
    if output_filename[-4:] != ".ply":
        raise ValueError(f"Incorrect filename {output_filename} must end with '.ply'")
    if len(points)==0:
        raise ValueError("No points to create the file")
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
    lines = pg.open_file_and_give_content(map_file)

    # Extraire les coordonnées XYZ
    points = [list(map(float, line.strip().split())) for line in lines]

    # Convertir la liste en tableau NumPy
    points_np = np.array(points, dtype=np.float32)

    save_ply_file(ply_file,points_np)

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

def centering_ply_on_mean_points(input_filename:str, output_filename:str):
    points,colors=get_points_and_colors_of_ply(input_filename)
    new_points=pa.centering_3Darray_on_mean_points(points)
    save_ply_file(output_filename,new_points,colors)

def color_ply_depending_on_axis(name_ply:str,new_name:str,axis:str):
    points,_=get_points_and_colors_of_ply(name_ply)
    colors=pa.color_3D_array_depending_on_axis(points,axis)
    save_ply_file(new_name,points,colors)

def remove_points_of_ply_below_threshold(input_ply_file: str, output_ply_file: str,threshold: float,axis:str):
    points,colors=get_points_and_colors_of_ply(input_ply_file)
    new_points,new_colors=pa.remove_points_of_array_below_threshold(points,threshold,colors,axis)   
    save_ply_file(output_ply_file,new_points,new_colors)

def reduce_density_of_ply(input_filename:str, output_filename:str,density:float):
    points,colors=get_points_and_colors_of_ply(input_filename)
    new_points,new_colors=pa.reduce_density_of_array(points,density,colors)
    save_ply_file(output_filename,new_points,new_colors)

def crop_ply_from_pixels_selection(input_filename:str, output_filename:str,shape:(int,int)):
    points,colors=get_points_and_colors_of_ply(input_filename)
    if len(colors)==0:
        raise ValueError(f"No image to display in the ply {input_filename} : the list of colors is empty")
    results=pa.crop_pc_from_zone_selection(points,colors,shape)
    save_ply_file(output_filename,results[0],results[1])
    
def filter_array_with_sphere_on_barycentre(input_filename:str, output_filename:str,radius:float):
    points,colors=get_points_and_colors_of_ply(input_filename)
    new_points,new_colors=pa.filter_array_with_sphere_on_barycentre(points,radius,colors)
    save_ply_file(output_filename,new_points,new_colors)
    
def apply_hsv_mask_to_ply(input_filename:str, output_filename:str,maskhsv: List[List[int]]):
    points,colors=get_points_and_colors_of_ply(input_filename)
    new_points,new_colors=pa.apply_hsv_mask_to_arrays(points,colors,maskhsv)
    save_ply_file(output_filename,new_points,new_colors)

def center_ply_on_image(input_filename:str, output_filename:str,image_target_path:str,shape:Tuple[int,int]=[]):
    points,colors=get_points_and_colors_of_ply(input_filename)
    new_points,new_colors=pa.center_pc_on_image(points,colors,image_target_path,shape)
    save_ply_file(output_filename,new_points,new_colors)

if __name__ == '__main__':
    # filter_array_with_sphere_on_barycentre("test.ply","test_barycentre.ply",0.06)
    center_ply_on_image("test.ply","test_centered.ply","image_ref.png",(640,480))
    # crop_ply_from_pixels_selection("test.ply","test_cropped.ply",(640,480))
    # points,colors=get_points_and_colors_of_ply('test.ply')
    # reduce_density_of_ply("test_colore.ply","test_reduce.ply",0.5)
    # centering_ply_on_mean_points("test.ply","test_centered.ply")
    # color_ply_depending_on_axis("test.ply","test_colore.ply","z")
    # remove_points_of_ply_below_threshold(10,"test_colore.ply","test2.ply")
    # points,colors=get_points_and_colors_of_ply('test.ply')
    # save_ply_file("with_colors.ply",points,colors)
    # save_ply_file("without_colors.ply",points)
    # save_ply_from_map("test.map","ply_from_map.ply")