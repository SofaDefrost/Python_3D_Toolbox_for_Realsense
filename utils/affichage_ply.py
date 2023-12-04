import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

def read_ply(file_path: str) -> np.ndarray:
    """
    Read a PLY file and extract vertices.

    Parameters:
    - file_path (str): The path to the PLY file.

    Returns:
    - np.ndarray: An array containing vertices from the PLY file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start = lines.index('end_header\n') + 1
    data = [list(map(float, line.strip().split())) for line in lines[data_start:]]
    vertices = np.array(data)

    return vertices

def plot_ply(vertices: np.ndarray) -> None:
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
        colors = vertices[:, 3:6] / 255.0  # Normalize colors to the range [0, 1]
    else:
        colors = 'b'  # Use blue as default color if no color information is present

    # Perform Delaunay triangulation
    triangles = Delaunay(vertices[:, :2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices with colors
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=colors, marker='o')

    # Plot the triangles
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles.simplices, color='gray', alpha=0.5)

    plt.show()

