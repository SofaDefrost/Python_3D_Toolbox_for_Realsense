import numpy as np

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from typing import List, Optional, Tuple

import processing_float as pfl


def plot_3D_array(vertices: np.ndarray) -> None:
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


def give_min_mean_max_of_list(list: List[float]) -> Optional[Tuple[float, float, float]]:
    """
    Calcule la valeur maximale, minimale et moyenne à partir d'une liste.

    Args:
        list (list): Liste des valeurs à analyser.

    Returns:
        tuple or None: Un tuple contenant la valeur minimale, moyenne et maximale, 
                      ou None si la liste est vide.
    """
    if not list:
        return None  # Retourne None si la liste est vide

    list_max = max(list)
    list_min = min(list)
    list_mean = sum(list) / len(list)

    return list_min, list_mean, list_max


def color_3D_array_depending_on_axis(points: np.ndarray, axis: str) -> np.ndarray:
    
    if axis != "x" and axis != "y" and axis != "z":
        raise ValueError(f"Unknown axis: {axis}. Must be x, y or z")
    
    colors = np.array([[0, 0, 0] for i in range(len(points))])
    
    if axis == "x":
        axis_int=0
    if axis == "y":
        axis_int=1
    if axis == "z":
        axis_int=2
    
    axis_coordinates = [coord[axis_int] for coord in points]
    min_coord, mean_coord, max_coord = give_min_mean_max_of_list(
        axis_coordinates)

    for i in range(len(colors)):
        colors[i] = pfl.convert_float_in_range_to_rgb(
            points[i][axis_int], min_coord, mean_coord, max_coord)
    return np.array(colors)


if __name__ == '__main__':
    print(give_min_mean_max_of_list([1.0, 2.0, 3.0]))
