import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from typing import List, Optional, Tuple

mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from .utils import array as array
    from .utils import float as f
    from . import processing_pixel_list as pixel
else:
    # Code executed as a script
    import utils.array as array
    import utils.float as f
    import processing_pixel_list as pixel


def plot(points: np.ndarray) -> None:
    """
    Plot a 3D array of points with optional color information.

    Parameters:
    - points (np.ndarray): The input array of points with shape (N, 3) or (N, 6) where the last three columns represent RGB color values.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    array.is_homogenous_of_dim(points, 3)
    # Extract color information if available
    if points.shape[1] == 6:
        # Normalize colors to the range [0, 1]
        colors = points[:, 3:6] / 255.0
    else:
        colors = 'b'  # Use blue as default color if no color information is present

    # Perform Delaunay triangulation
    triangles = Delaunay(points[:, :2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices with colors
    ax.scatter(points[:, 0], points[:, 1],
               points[:, 2], c=colors, marker='o')

    # Plot the triangles
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                    triangles=triangles.simplices, color='gray', alpha=0.5)

    plt.show()


def give_max_distance(points: np.ndarray) -> float:
    """
    Calculate the maximum pairwise distance between points in a 3D point cloud.

    Parameters:
    - points (np.ndarray): The 3D coordinates of the point cloud.

    Returns:
    float: The maximum pairwise distance between points.
    """
    # Compute every pairs of distances
    distances_paires = pdist(points)

    # Convert pairs in a matrix
    distances_matrice = squareform(distances_paires)

    # Find the max of this matrix
    distance_max = np.max(distances_matrice)

    return distance_max


def get_color_depending_on_axis(points: np.ndarray, axis: str) -> np.ndarray:
    """
    Get colors for a 3D array based on the values along a specific axis.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - axis (str): The axis along which to determine colors ('x', 'y', or 'z').

    Returns:
    - np.ndarray: The array of RGB colors corresponding to the values along the specified axis.

    Raises:
    - ValueError: If the input array is not of the correct shape or if the axis is unknown.
    """
    array.is_homogenous_of_dim(points, 3)

    if axis != "x" and axis != "y" and axis != "z":
        raise ValueError(f"Unknown axis: {axis}. Must be x, y or z")

    colors = np.array([[0, 0, 0] for i in range(len(points))])

    if axis == "x":
        axis_int = 0
    if axis == "y":
        axis_int = 1
    if axis == "z":
        axis_int = 2

    axis_coordinates = [coord[axis_int] for coord in points]
    min_coord, mean_coord, max_coord = array.give_min_mean_max(
        axis_coordinates)

    for i in range(len(colors)):
        colors[i] = f.convert_float_in_range_to_rgb(
            points[i][axis_int], min_coord, mean_coord, max_coord)
    return np.array(colors)


def get_center_geometry(points: np.ndarray) -> List[float]:
    """
    Calculate the center coordinates of a set of 3D points.

    Args:
        points (np.ndarray): Input array of 3D coordinates.

    Returns:
        List[float]: Center coordinates [x, y, z].
    """
    coord_x = [point[0] for point in points]
    coord_y = [point[1] for point in points]
    coord_z = [point[2] for point in points]
    max_x = max(coord_x)
    min_x = min(coord_x)
    max_y = max(coord_y)
    min_y = min(coord_y)
    max_z = max(coord_z)
    min_z = min(coord_z)
    return [(max_x+min_x)/2, (max_y+min_y)/2, (max_z+min_z)/2]


def centers_points_on_geometry(points: np.ndarray) -> np.ndarray:
    """
    Center a set of 3D points around their mean.

    Args:
        points (np.ndarray): Input array of 3D coordinates.

    Returns:
        np.ndarray: Centered 3D points.
    """
    mean_point = get_center_geometry(points)
    new_vertices = [point - mean_point for point in points]
    return np.array(new_vertices)


def get_mean_point(points: np.ndarray) -> [float, float, float]:
    """
    Calculate the mean point of a 3D array.

    Parameters:
    - points (np.ndarray): The 3D array containing points.

    Returns:
    np.ndarray: The mean point [mean_x, mean_y, mean_z].
    """
    array.is_homogenous_of_dim(points, 3)
    tab_x = []
    tab_y = []
    tab_z = []

    for i in points:
        tab_x.append(i[0])
        tab_y.append(i[1])
        tab_z.append(i[2])

    middle_x = np.mean(tab_x)
    middle_y = np.mean(tab_y)
    middle_z = np.mean(tab_z)

    return [middle_x, middle_y, middle_z]


def centering_on_mean_points(points: np.ndarray) -> np.ndarray:
    """
    Center a 3D array of points around their mean.

    Parameters:
    - array (np.ndarray): The input array of 3D points.

    Returns:
    - np.ndarray: The array of 3D points centered around their mean.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """

    pt_middle = get_mean_point(points)

    new_vertices = [point - pt_middle for point in points]

    return np.array(new_vertices)


def remove_points_threshold(points: np.ndarray, threshold: float, position: int = 1, colors: Optional[np.ndarray] = [], axis: str = "z") -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove points from a 3D array based on a threshold along a specific axis.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - threshold (float): The threshold value for removing points.
    - position (int): 1 for above, -1 for below
    - colors (np.ndarray, optional): The array of colors associated with the points.
    - axis (str, optional): The axis along which to apply the threshold ('x', 'y', or 'z').

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The filtered array of 3D points (and colors if provided).

    Raises:
    - ValueError: If the input array is not of the correct shape or if the axis is unknown.
    """

    if (position != -1) and (position != 1):
        raise ValueError(
            f"Incorrect position, given {position} expected 1 or -1.")

    array.is_homogenous_of_dim(points, 3)

    if axis != "x" and axis != "y" and axis != "z":
        raise ValueError(f"Unknown axis: {axis}. Must be x, y or z")

    if axis == "x":
        axis_int = 0
    if axis == "y":
        axis_int = 1
    if axis == "z":
        axis_int = 2

    if position == -1:
        # Select indices of points whose coordinate along the specified axis is greater than or equal to the threshold
        index = np.where(points[:, axis_int] >= threshold)
    else:
        index = np.where(points[:, axis_int] <= threshold)

    # Check if the filtered point cloud is not empty
    if len(index[0]) == 0:
        raise ValueError("Every points have been removed")

    new_colors = []

    if len(colors) > 0:
        new_colors = colors[index]

    new_points = points[index]
    return np.array(new_points), np.array(new_colors)


def reduce_density(points: np.ndarray, density: float, colors: Optional[np.ndarray] = []) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce the density of a 3D point cloud by randomly selecting a fraction of the points.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - density (float): The fraction of points to keep in the range (0, 1).
    - colors (np.ndarray, optional): The array of colors associated with the points.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The reduced array of 3D points and colors if provided.

    Raises:
    - ValueError: If the input array is not of the correct shape or if the density is outside the valid range.
    """
    array.is_homogenous_of_dim(points, 3)

    if not (0 < density <= 1):
        raise ValueError("Density must be in the range (0, 1]")

    colors_reduced = []

    # Calculate the number of points to keep
    number_points = int(len(points) * density)

    # Randomly select indices of points to keep
    indices_to_keep = np.random.choice(
        len(points), number_points, replace=False)

    # Extract the selected points
    points_reduits = points[indices_to_keep, :]
    if len(colors) > 0:
        colors_reduced = colors[indices_to_keep, :]

    return np.array(points_reduits), np.array(colors_reduced)


def filter_with_sphere_on_barycentre(points: np.ndarray, radius: float, colors: Optional[np.ndarray] = [], tab_index: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter a 3D point cloud by keeping only the points within a sphere centered at the barycentre.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - radius (float): The radius of the sphere.
    - colors (np.ndarray, optional): The array of colors associated with the points.
    - indices (List[int], optional): The array of indices associated with the points.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]: The filtered array of 3D points and colors and indices (if provided).

    Raises:
    - ValueError: If the input array is not of the correct shape or if the radius is negative.
    """
    array.is_homogenous_of_dim(points, 3)
    if radius < 0:
        raise ValueError("Radius must be non-negative")

    barycentre = np.mean(points, axis=0)
    radius_filter = radius
    filtered_points = []
    filtered_colors = []
    filtered_indice = []

    for i, point in enumerate(points):
        distance = np.linalg.norm(point - barycentre)
        if distance <= radius_filter:
            filtered_points.append(point)
            if len(colors) > 0:
                filtered_colors.append(colors[i])
            if len(tab_index) > 0:
                filtered_indice.append(tab_index[i])

    return np.array(filtered_points), np.array(filtered_colors), np.array(filtered_indice)


def resize_with_scaling_factor(points: np.ndarray, scaling_factor: float) -> np.ndarray:
    """
    Resize a point cloud by applying a scaling factor to each point.

    Parameters:
    - points (np.ndarray): The input point cloud represented as a 2D numpy array.
    - scaling_factor (float): The scaling factor to be applied to the point cloud.

    Returns:
    np.ndarray: The resized point cloud.
    """
    # Apply scaling to points
    scaled_points = points * scaling_factor

    return np.array(scaled_points)


def get_scaling_factor_between_point_cloud(points_input: np.ndarray, points_reference: np.ndarray) -> float:
    """
    Calculate the scaling factor between two point clouds based on their maximum distances.

    Args:
    - points_input (np.ndarray): Input point cloud.
    - points_reference (np.ndarray): Reference point cloud.

    Returns:
    - float: Scaling factor.
    """
    # Check if the point clouds are not empty
    if len(points_input) < 2 or len(points_reference) < 2:
        raise ValueError(
            "Point clouds must have at least two points for scaling factor calculation.")

    # Calculate the maximum distances of the two point clouds
    max_dist_pc_input = give_max_distance(points_input)
    max_dist_pc_ref = give_max_distance(points_reference)

    # Calculate the scaling factor based on the maximum distances
    return max_dist_pc_ref / max_dist_pc_input


def resize_point_cloud_to_another_one(points_input: np.ndarray, points_reference: np.ndarray) -> np.ndarray:
    """
    Resize a point cloud to match the scale of another point cloud.

    Parameters:
    - points_input (np.ndarray): The input point cloud to be resized.
    - points_reference (np.ndarray): The reference point cloud with the desired scale.

    Returns:
    np.ndarray: The resized point cloud.
    """
    scaling_factor = get_scaling_factor_between_point_cloud(
        points_input, points_reference)
    return resize_with_scaling_factor(points_input, scaling_factor)


def crop_from_zone_selection(points: np.ndarray, colors: np.ndarray, shape: Tuple[int, int] = [], tab_index: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Crop a region of interest (ROI) from a point cloud based on user selection.

    Args:
        points (np.ndarray): Input point cloud coordinates.
        colors (np.ndarray): Input point cloud colors.
        shape (Tuple[int, int], optional): Shape of the display window. Defaults to [].
        tab_index (Optional[List[int]], optional): List of indices for the point cloud. Defaults to [].

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]: Cropped point cloud, colors, indices, and new shape.
    """
    # Global variables to store information of mouseclick
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    cropping = False

    def mouse_click(event, x, y, flags, param):

        nonlocal start_x, start_y, end_x, end_y, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start cropping
            start_x, start_y = x, y
            end_x, end_y = x, y
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            # End cropping
            end_x, end_y = x, y
            cropping = False
            # Draw selection
            cv2.rectangle(colors_image, (start_x, start_y),
                          (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow("Cropping", colors_image[:, :, ::-1])
    if shape != []:
        if np.shape(shape) != (2,):
            raise ValueError(f"Incorrect shape {shape} for the display")
        if shape[0] == np.shape(colors)[0] and shape[1] == np.shape(colors)[1] and 3 == np.shape(colors)[2]:
            colors_image = colors.astype(np.uint8)
            colors = array.to_line(colors)
            height, length = np.shape(colors_image)[
                0], np.shape(colors_image)[1]
        else:
            height, length = shape
            colors_image = array.line_to_2Darray(
                colors, (length, height)).astype(np.uint8)
            colors = array.to_line(colors)
    else:
        if len(np.shape(colors)) != 3:
            raise ValueError(
                f"Incorrect shape for the display got {np.shape(colors)} and expected (x,y,z)")
        if np.shape(colors)[2] != 3:
            raise ValueError(
                f"Incorrect dimension for the array, expected 3 and given {np.shape(colors)[2]}")
        colors_image = colors.astype(np.uint8)
        colors = array.to_line(colors)
        length, height = np.shape(colors_image)[0], np.shape(colors_image)[1]

    # Create a window for the display
    cv2.namedWindow("Cropping")
    cv2.setMouseCallback("Cropping", mouse_click)

    # User's instructions
    print("Use the mouse to select the cropping rectangle. Press the 'q' to finish cropping.")

    while True:
        cv2.imshow("Cropping", colors_image[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF

        # Leave the programm if key 'c'
        if key == ord("q"):
            break

    # Check if the selection has a valid shape
    if start_x == end_x or start_y == end_y:
        raise ValueError("Incorrect values for cropping selected")

    # Get the data for the cropping
    x_min, y_min = min(start_x, end_x), min(start_y, end_y)
    x_max, y_max = max(start_x, end_x), max(start_y, end_y)

    # Filtering of the points
    bottom_left_corner = (y_min-1)*height + x_min
    top_left_corner = (y_max-1)*height + x_min
    bottom_right_corner = (y_min-1)*height + x_max

    new_shape = (abs(x_max-x_min), abs(y_max-y_min)+1)

    i = 0
    points_cloud_crop = []
    colors_crop = []
    tab_index_crop = []

    while (bottom_left_corner != top_left_corner):
        for j in range(bottom_left_corner, bottom_right_corner):
            points_cloud_crop.append(points[j])
            colors_crop.append(colors[j])
            if len(tab_index) > 0:
                tab_index_crop.append(tab_index[j])
        bottom_left_corner = (y_min+i-1)*height + x_min
        bottom_right_corner = (y_min+i-1)*height + x_max
        i += 1

    cv2.destroyAllWindows()
    return np.array(points_cloud_crop), np.array(colors_crop), np.array(tab_index_crop), new_shape


def apply_binary_mask(points: np.ndarray, colors: np.ndarray, mask: np.ndarray, shape: Tuple[int, int], tab_index: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a binary mask to filter points and colors.

    Args:
        points (np.ndarray): Input array of 3D coordinates.
        colors (np.ndarray): Input array of colors.
        mask (np.ndarray): Binary mask to apply.
        shape (Tuple[int, int]): Shape of the original data.
        tab_index (Optional[List[int]]): Optional array of indices.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered points, colors, and optional indices.
    """
    colors = cv2.convertScaleAbs(colors)
    cv2.normalize(colors, colors, 0, 255, cv2.NORM_MINMAX)
    colors = colors.astype(np.uint8)
    colors_3D = array.line_to_2Darray(colors, (shape[1], shape[0]))
    # Mask application
    colors_filter = colors_3D[mask]

    points_line = array.line_to_2Darray(points, (shape[1], shape[0]))
    points_filter = points_line[mask]
    points_filter = array.to_line(points_filter)

    if len(tab_index) > 0:
        tab_index_line = tab_index.reshape((shape[1], shape[0], 1))
        tab_index_filter = tab_index_line[mask]
        tab_index_filter = tab_index_filter.reshape(
            (np.shape(tab_index_filter)[0]*np.shape(tab_index_filter)[1], 1))
    else:
        tab_index_filter = []

    return np.array(points_filter), np.array(colors_filter), np.array(tab_index_filter)


def apply_hsv_mask(points: np.ndarray, colors: np.ndarray, maskhsv: Tuple[np.ndarray, np.ndarray], shape: Tuple[int, int], tab_index: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply an HSV mask to filter points and colors.

    Args:
        points (np.ndarray): Input array of 3D coordinates.
        colors (np.ndarray): Input array of colors.
        maskhsv (Tuple[np.ndarray, np.ndarray]): Tuple containing lower and upper HSV values.
        shape (Tuple[int, int]): Shape of the original data.
        tab_index (Optional[List[int]]): Optional array of indices.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered points, colors, and indices.
    """
    colors = cv2.convertScaleAbs(colors)
    cv2.normalize(colors, colors, 0, 255, cv2.NORM_MINMAX)
    colors = colors.astype(np.uint8)
    colors_3D = array.line_to_2Darray(colors, (shape[1], shape[0]))

    # Mask reconstruction
    lower_hsv, upper_hsv = maskhsv
    image = colors_3D[:, :, ::-1]  # Conversion RGB to BGR
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    binary_mask = mask > 0

    return apply_binary_mask(points, colors, binary_mask, shape, tab_index)


def center_on_image(points: np.ndarray, colors: np.ndarray, shape_pc: Tuple[int, int], image_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center a point cloud on the target image.

    Parameters:
    - points (np.ndarray): The array of 3D points.
    - colors (np.ndarray): The array of colors associated with the points.
    - shape (Tuple[int, int], optional): The shape of the points cloud.
    - image_target (np.ndarray): The array of target image.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The centered array of 3D points and colors.
    """
    # Get the size of the target image

    height, width = image_target.shape[:2]

    # Calculate the center of the image in pixel coordinates
    center_image_ref = np.array(
        [int(width/2), int(height/2), 1])  # In pixel

    image_source = array.line_to_2Darray(colors, [shape_pc[1],shape_pc[0]])
    image_source = cv2.convertScaleAbs(image_source)
    # Get the size of the source image
    h, w = image_source.shape[:2]
    # Calculate the coordinates of the center in the source image (in pixel coordinates)
    homography_matrix = pixel.get_homography(
        image_target, image_source)

    projected_center = np.dot(homography_matrix, center_image_ref)  # in pixel

    # Convert to depth coordinates
    pc_original = points[(round(projected_center[1])-1)
                         * w + int(projected_center[0])]

    # Reposition the point cloud to be centered around the origin point
    pc_centered = [(x[0] - pc_original[0], x[1] -
                    pc_original[1], x[2] - pc_original[2]) for x in points]

    return np.array(pc_centered), colors


def find_nearest_neighbors_between_pc(source_points: np.ndarray, target_points: np.ndarray, nearest_neigh_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest neighbors in the source point cloud for each point in the target point cloud.

    Args:
    - source_points (np.ndarray): Source point cloud.
    - target_points (np.ndarray): Target point cloud.
    - nearest_neigh_num (int): Number of nearest neighbors to find.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Nearest neighbor points and their indices in the source point cloud.
    """
    # Create a KD-Tree from the source points
    source_kdtree = cKDTree(source_points)

    # Find nearest neighbors for each point in the target point cloud
    _, indices = source_kdtree.query(
        target_points, k=nearest_neigh_num)

    # Retrieve the nearest neighbor points

    nearest_neighbors = source_points[indices]

    return nearest_neighbors, np.array(indices)
