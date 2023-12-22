import numpy as np
import open3d as o3d
import logging
import sys

from typing import List, Tuple, Optional

mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from . import processing_general as pg
    from . import processing_array as pa
    from . import display_function_Tkinter as tk
else:
    # Code executed as a script
    import processing_general as pg
    import processing_array as pa
    import display_function_Tkinter as tk


def plot_ply(input_filename: str) -> None:
    """
    Plot 3D points from PLY file.

    Parameters:
    - path_name (string): The name of the ply to plot

    Returns:
    - None
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    vertices = [point + color for point, color in zip(points, colors)]
    pa.plot_3D_array(np.array(vertices))


def save_ply_file(output_filename: str, points: np.ndarray, colors: Optional[np.ndarray] = []) -> None:
    """
    Save 3D points and optionally colors to a PLY file.

    Parameters:
    - output_filename (str): The name of the output PLY file.
    - points (np.ndarray): Array of 3D points, where each row represents a point.
    - colors (Optional[np.ndarray], optional): Array of RGB colors corresponding to each point.
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


def save_ply_from_map(map_file: str, ply_file: str) -> None:
    """
    Read coordinates from a map file and save them to a PLY file.

    Parameters:
    - map_file (str): The path to the input map file.
    - ply_file (str): The name of the output PLY file.

    Returns:
    - None: The function does not return anything, but it creates a PLY file with the specified data.
    """
    lines = pg.open_file_and_give_content(map_file)

    # Extract XYZ coordinates
    points = [list(map(float, line.strip().split())) for line in lines]

    # Convert the list to a NumPy array
    points_np = np.array(points, dtype=np.float32)

    save_ply_file(ply_file, points_np)


def get_points_and_colors_of_ply(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a .ply file to get a point cloud with RGB colors in the range of 0 to 255.

    Parameters:
    - file_path (str): Path to the .ply file.

    Returns:
    - tuple: A tuple containing a NumPy array of points and a NumPy array of colors.
    """
    # Guaranteed that the file exists
    pg.open_file_and_give_content(file_path)

    # Load data from the .ply file
    ply_data = o3d.io.read_point_cloud(file_path)

    # Retrieve points and colors
    points = np.array(ply_data.points)
    colors = np.array(ply_data.colors) * 255

    return points, colors


def centering_ply_on_mean_points(input_filename: str, output_filename: str) -> None:
    """
    Center a PLY file based on the mean of its points and save the result to a new file.

    Parameters:
    - input_filename (str): The path to the input PLY file.
    - output_filename (str): The name of the output PLY file after centering.

    Returns:
    - None: The function does not return anything, but it creates a centered PLY file.
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    new_points = pa.centering_3Darray_on_mean_points(points)
    save_ply_file(output_filename, new_points, colors)


def color_ply_depending_on_axis(input_filename: str, output_filename: str, axis: str) -> None:
    """
    Color a PLY file based on the specified axis and save the result to a new file.

    Parameters:
    - name_ply (str): The path to the input PLY file.
    - new_name (str): The name of the output PLY file after coloring.
    - axis (str): The axis along which to color the points ('x', 'y', or 'z').

    Returns:
    - None: The function does not return anything, but it creates a colored PLY file.
    """
    points, _ = get_points_and_colors_of_ply(input_filename)
    colors = pa.get_color_3D_array_depending_on_axis(points, axis)
    save_ply_file(output_filename, points, colors)


def remove_points_of_ply_below_threshold(input_filename: str, output_filename: str, threshold: float, axis: str) -> None:
    """
    Remove points from a PLY file below a specified threshold along a given axis and save the result to a new file.

    Parameters:
    - input_filename (str): The path to the input PLY file.
    - output_filename (str): The name of the output PLY file after removing points.
    - threshold (float): The threshold value below which points will be removed.
    - axis (str): The axis along which to apply the threshold ('x', 'y', or 'z').

    Returns:
    - None: The function does not return anything, but it creates a PLY file with points removed below the threshold.
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    new_points, new_colors = pa.remove_points_of_array_below_threshold(
        points, threshold, colors, axis)
    save_ply_file(output_filename, new_points, new_colors)


def remove_points_of_ply_below_threshold_with_interface(input_filename: str, output_filename: str) -> None:
    """
    Remove points from a PLY file that are below a certain threshold using a Tkinter interface.

    Parameters:
    - input_filename (str): Path to the input PLY file.
    - output_filename (str): Path to save the output PLY file.

    Returns:
    None
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    threshold = tk.get_parameter_function_on_array_Tkinter(
        points, pa.remove_points_of_array_below_threshold)
    new_points, new_colors = pa.remove_points_of_array_below_threshold(
        points, threshold, colors)
    save_ply_file(output_filename, new_points, new_colors)


def reduce_density_of_ply(input_filename: str, output_filename: str, density: float) -> None:
    """
    Reduce the density of points in a PLY file and save the result to a new file.

    Parameters:
    - input_filename (str): The path to the input PLY file.
    - output_filename (str): The name of the output PLY file after density reduction.
    - density (float): The target density of points after reduction, given as a fraction.

    Returns:
    - None: The function does not return anything, but it creates a PLY file with reduced point density.
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    new_points, new_colors = pa.reduce_density_of_array(
        points, density, colors)
    save_ply_file(output_filename, new_points, new_colors)


def reduce_density_of_ply_with_interface(input_filename: str, output_filename: str) -> None:
    """
    Reduce the density of a PLY file using a Tkinter interface.

    Parameters:
    - input_filename (str): Path to the input PLY file.
    - output_filename (str): Path to save the output PLY file.

    Returns:
    None
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    density = tk.get_parameter_function_on_array_Tkinter(
        points, pa.reduce_density_of_array)
    new_points, new_colors = pa.reduce_density_of_array(
        points, density, colors)
    save_ply_file(output_filename, new_points, new_colors)


def filter_array_with_sphere_on_barycentre(input_filename: str, output_filename: str, radius: float) -> None:
    """
    Filter points in a PLY file based on a sphere centered at the barycenter and save the result to a new file.

    Parameters:
    - input_filename (str): The path to the input PLY file.
    - output_filename (str): The name of the output PLY file after filtering.
    - radius (float): The radius of the sphere used for filtering.

    Returns:
    - None: The function does not return anything, but it creates a PLY file with points filtered by a sphere.
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    new_points, new_colors = pa.filter_array_with_sphere_on_barycentre(
        points, radius, colors)
    save_ply_file(output_filename, new_points, new_colors)


def filter_array_with_sphere_on_barycentre_with_interface(input_filename: str, output_filename: str) -> None:
    """
    Filter points from a PLY file using a sphere centered on the barycenter with a Tkinter interface.

    Parameters:
    - input_filename (str): Path to the input PLY file.
    - output_filename (str): Path to save the output PLY file.

    Returns:
    None
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    radius = tk.get_parameter_function_on_array_Tkinter(
        points, pa.filter_array_with_sphere_on_barycentre)
    new_points, new_colors = pa.filter_array_with_sphere_on_barycentre(
        points, radius, colors)
    save_ply_file(output_filename, new_points, new_colors)


def crop_ply_from_pixels_selection(input_filename: str, output_filename: str, shape: Tuple[int, int]) -> None:
    """
    Crop a PLY file based on a pixel selection from an associated image and save the result to a new file.

    Parameters:
    - input_filename (str): The path to the input PLY file.
    - output_filename (str): The name of the output PLY file after cropping.
    - shape (Tuple[int, int]): The shape of the point cloud associated to the PLY file.

    Raises:
    - ValueError: If there is no image to display in the PLY file (the list of colors is empty).

    Returns:
    - None: The function does not return anything, but it creates a cropped PLY file.
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    if len(colors) == 0:
        raise ValueError(
            f"No image to display in the ply {input_filename} : the list of colors is empty")
    results = pa.crop_pc_from_zone_selection(points, colors, shape)
    save_ply_file(output_filename, results[0], results[1])


def apply_hsv_mask_to_ply(input_filename: str, output_filename: str, maskhsv: Tuple[np.ndarray, np.ndarray]) -> None:
    """
    Apply an HSV mask to the colors of a PLY file and save the result to a new file.

    Parameters:
    - input_filename (str): The path to the input PLY file.
    - output_filename (str): The name of the output PLY file after applying the HSV mask.
    - maskhsv (List[List[int]]): The HSV values defining the color mask.

    Returns:
    - None: The function does not return anything, but it creates a PLY file with colors filtered by the HSV mask.
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    new_points, new_colors = pa.apply_hsv_mask_to_pc(
        points, colors, maskhsv)
    save_ply_file(output_filename, new_points, new_colors)


def center_ply_on_image(input_filename: str, output_filename: str, image_target_path: str, shape: Tuple[int, int] = []) -> None:
    """
    Center a PLY file based on an associated image and save the result to a new file.

    Parameters:
    - input_filename (str): The path to the input PLY file.
    - output_filename (str): The name of the output PLY file after centering.
    - image_target_path (str): The path to the target image used for centering.
    - shape (Tuple[int, int], optional): The shape of the point cloud associated to the PLY file.

    Returns:
    - None: The function does not return anything, but it creates a centered PLY file.
    """
    points, colors = get_points_and_colors_of_ply(input_filename)
    new_points, new_colors = pa.center_pc_on_image(
        points, colors, image_target_path, shape)
    save_ply_file(output_filename, new_points, new_colors)


if __name__ == '__main__':
    # filter_array_with_sphere_on_barycentre("./example/test.ply","./example/test_barycentre.ply",0.06)
    # crop_ply_from_pixels_selection("./example/test.ply","./example/test_cropped.ply",(640,480))
    # reduce_density_of_ply("./example/test.ply","./example/test_reduce.ply",0.5)
    # centering_ply_on_mean_points("./example/test.ply","./example/test_centered.ply")
    # color_ply_depending_on_axis("./example/test.ply","./example/test_colored_y.ply","y")
    # remove_points_of_ply_below_threshold("test_colore.ply","test_below_threshold.ply",0.1,"z")
    # points, colors = get_points_and_colors_of_ply('./example/test.ply')
    # save_ply_file("./example/test_with_colors.ply", points, colors)
    # save_ply_file("./example/test_without_colors.ply", points)
    # save_ply_from_map("test.map","ply_from_map.ply")
    center_ply_on_image("./example/capture_with_image_ref.ply", "./example/capture_with_image_ref_centred.ply",
                        "./example/image_ref.png", (640, 480))
