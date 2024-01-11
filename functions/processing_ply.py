import numpy as np
import open3d as o3d
import logging
import sys
import pymeshlab

from typing import Tuple, Optional

mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from .utils import files as file
else:
    # Code executed as a script
    import utils.files as file


def save(output_filename: str, points: np.ndarray, colors: Optional[np.ndarray] = []) -> None:
    """
    Save 3D points and optionally colors to a PLY file.

    Parameters:
    - output_filename (str): The name of the output PLY file.
    - points (np.ndarray): Array of 3D points, where each row represents a point.
    - colors (np.ndarray, optional): Array of RGB colors corresponding to each point.
      If not provided, the PLY file will only contain point coordinates.

    Raises:
    - ValueError: If the output_filename is incorrect, does not end with '.ply',
      or if there are no points to create the file.
    - ValueError: If the number of points does not match the number of colors when colors are provided.

    Returns:
    - None: The function does not return anything, but it creates a PLY file with the specified data.
    """
    if len(output_filename) < 5:
        raise ValueError(f"Incorrect filename {output_filename}")
    if output_filename[-4:] != ".ply":
        raise ValueError(
            f"Incorrect filename {output_filename} must end with '.ply'")
    if len(points) == 0:
        raise ValueError("No points to create the file")
    with_color = False
    if len(colors) > 0:
        with_color = True

    if (len(points) != len(colors)) and with_color:
        raise ValueError(
            "The number of points must match the number of colors.")

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

        logging.info(f"The file '{output_filename}' has been created.")


def create_and_save_mesh_from_ply(ply_filename: str, obj_filename: str, number_neighbors: Optional[int] = 100, reconstruction_depth: Optional[int] = 16) -> None:
    """
    Create and save a mesh in OBJ format from a PLY file using MeshLab commands.

    Parameters:
    - ply_filename (str): The path to the input PLY file.
    - obj_filename (str): The path to save the output OBJ file. It must end with '.obj'.
    - number_neighbors (int): The number of neighbors used in computing normals for point clouds (default: 100).
    - reconstruction_depth (int): The depth parameter for surface reconstruction (default: 16).

    Returns:
    - None
    """
    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load mesh
    ms.load_new_mesh(ply_filename)
    ms.compute_normal_for_point_clouds(k=number_neighbors)
    ms.generate_surface_reconstruction_screened_poisson(
        depth=reconstruction_depth)

    # save the current selected mesh
    ms.save_current_mesh(obj_filename)


def create_and_save_ply_from_map(map_file: str, ply_file: str) -> None:
    """
    Read coordinates from a map file and save them to a PLY file.

    Parameters:
    - map_file (str): The path to the input map file.
    - ply_file (str): The name of the output PLY file.

    Returns:
    - None: The function does not return anything, but it creates a PLY file with the specified data.
    """
    lines = file.open_and_give_content(map_file)

    # Extract XYZ coordinates
    points = [list(map(float, line.strip().split())) for line in lines]

    # Convert the list to a NumPy array
    points_np = np.array(points, dtype=np.float32)

    save(ply_file, points_np)


def get_points_and_colors(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a .ply file to get a point cloud with RGB colors in the range of 0 to 255.

    Parameters:
    - file_path (str): Path to the .ply file.

    Returns:
    - tuple: A tuple containing a NumPy array of points and a NumPy array of colors.
    """
    # Guaranteed that the file exists
    file.is_existing(file_path)

    # Load data from the .ply file
    ply_data = o3d.io.read_point_cloud(file_path)

    # Retrieve points and colors
    points = np.array(ply_data.points)
    colors = np.array(ply_data.colors) * 255

    return points, colors


if __name__ == '__main__':
    # points, colors = get_points_and_colors_of_ply('./example/test.ply')
    # save_ply_file("./example/test_with_colors.ply", points, colors)
    # save_ply_file("./example/test_without_colors.ply", points)
    # create_and_save_ply_from_map("test.map","ply_from_map.ply")
    create_and_save_mesh_from_ply("./example/test.ply", "./example/test.obj")
