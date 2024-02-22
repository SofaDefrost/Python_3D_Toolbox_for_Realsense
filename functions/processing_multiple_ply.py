import numpy as np
import sys

from typing import List

mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from .utils import array as array
else:
    # Code executed as a script
    import utils.array as array


def save(output_filename: str, list_pc: List) -> None:
    """
    Save point clouds data to an .mply file.

    Args:
    - output_filename (str): Name of the output file.
    - list_pc (List): List of point clouds to be saved.

    Raises:
    - ValueError: If the output filename is incorrect or if there are no point clouds to create the file.
    """
    if len(output_filename) < 6:
        raise ValueError(f"Incorrect filename {output_filename}")
    if output_filename[-5:] != ".mply":
        raise ValueError(
            f"Incorrect filename {output_filename} must end with '.mply'")
    number_of_pc = len(list_pc)
    if number_of_pc == 0:
        raise ValueError("No point cloud to create the file")

    if len(list_pc[0]) != 2:
        size_pc = list_pc[0]
    else:
        size_pc = len(list_pc[0][0])

    with open(output_filename, 'w') as mply_file:
        # Écriture de l'en-tête mPLY
        mply_file.write("mply\n")
        mply_file.write("format ascii 1.0\n")
        mply_file.write(f"number pc {number_of_pc}\n")
        # all the pc must have the same shape
        mply_file.write(f"size pc {size_pc}\n")
        mply_file.write("property float x\n")
        mply_file.write("property float y\n")
        mply_file.write("property float z\n")
        mply_file.write("property uchar red\n")
        mply_file.write("property uchar green\n")
        mply_file.write("property uchar blue\n")
        mply_file.write("end_header\n")

        for i in range(number_of_pc):
            pc = list_pc[i]
            if len(pc) != 2:
                points = pc
                # There are no colors
                for point in points:
                    x = point[0]
                    y = point[1]
                    z = point[2]
                    mply_file.write(f"{x} {y} {z} 0 0 0\n")
            else:
                points = pc[0]
                colors = pc[1]
                colors = array.to_line(colors)
                if (len(points) != len(colors)):
                    raise ValueError(
                        "The number of points must match the number of colors.")

                for point, color in zip(points, colors):
                    x = point[0]
                    y = point[1]
                    z = point[2]
                    r, g, b = color
                    mply_file.write(
                        f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

            mply_file.write(f"end_pc_{i}\n")


def get_point_cloud(file_path, pc_index):
    """
    Read point cloud data from an .mply file.

    Args:
    - file_path (str): Path to the .mply file.
    - pc_index (int): Index of the point cloud to retrieve.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Tuple containing points and colors (if available) of the specified point cloud.
    """
    points = []
    colors = []
    file = open(file_path, "r")
    line = file.readline()  # mply
    line = file.readline()  # format ascii 1.0
    number_pc = int(file.readline().strip().split()[-1:][0])  # number pc
    if pc_index > number_pc:
        raise ValueError(
            f"There is no {pc_index} point cloud in the file {file_path}")
    size_pc = int(file.readline().strip().split()[-1:][0])  # size pc
    while not (line.startswith("end_header")):
        line = file.readline()
    i = -1
    while i != (pc_index-1)*(size_pc+1):
        line = file.readline()
        i += 1
    for j in range(size_pc):
        data = line.strip().split()
        points.append([float(x) for x in data[:3]])
        if len(data) > 3:
            colors.append([int(x) for x in data[-3:]])
        line = file.readline()
    return np.array(points), np.array(colors)


if __name__ == '__main__':
    import processing_ply as ply
    list_pc=get_point_cloud("test.mply",239)
    ply.save("test.ply",list_pc[0],list_pc[1])
