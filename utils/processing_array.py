import numpy as np
import pyvista as pv
import cv2
import matplotlib.pyplot as plt
import sys

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from typing import List, Optional, Tuple

mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from . import processing_img as pi
    from . import processing_general as pg
    from . import processing_ply as pp
    from . import processing_float as pf
else:
    # Code executed as a script
    import processing_ply as pp
    import processing_float as pf
    import processing_img as pi
    import processing_general as pg

# Arrays


def is_homogenous_array_of_dim(array: np.ndarray, dimension: Optional[int] = -1) -> None:
    """
    Check if the input array is homogeneous and has the specified dimension.

    Parameters:
    - array (List[Union[int, float, str]]): The input array to be checked.
    - dimension (Optional[int]): The expected dimension of the array. If not specified, any dimension is allowed.

    Raises:
    - ValueError: If the array is empty, contains different types, or has an incorrect dimension.
    """
    if len(array) == 0:
        raise ValueError(f"Empty array {array}")
    if not all(isinstance(element, type(array[0])) for element in array):
        raise ValueError(f"Different type in the array {array}")
    if dimension > 0:
        taille = np.shape(array)
        if not (len(taille) == 1 and dimension == 1):
            if not taille[1] == dimension:
                raise ValueError(
                    f"Not the correct dimension for the array {array}, dimension expected {dimension}")


def plot_3D_array(vertices: np.ndarray) -> None:
    """
    Plot a 3D array of vertices with optional color information.

    Parameters:
    - vertices (np.ndarray): The input array of vertices with shape (N, 3) or (N, 6) where the last three columns represent RGB color values.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    is_homogenous_array_of_dim(vertices, 3)
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


def build_mesh_from_3Darray(points: np.ndarray) -> np.ndarray:
    """
    Build a mesh from a 3D array of points using Delaunay triangulation.

    Parameters:
    - points (np.ndarray): The input array of points with shape (N, 3).

    Returns:
    - np.ndarray: The list of triangles forming the mesh.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    is_homogenous_array_of_dim(points, 3)
    cloud = pv.PolyData(points)
    cloud.plot(point_size=15)

    surf = cloud.delaunay_2d()

    # Access the list of triangles
    triangles = surf.faces.tolist()

    liste_indice_triangles = []
    indice = 1
    while indice <= len(triangles)-3:
        liste_indice_triangles.append(
            [triangles[indice], triangles[indice+1], triangles[indice+2]])
        indice = indice + 4

    # Display the list of triangles

    # C'est la liste des triangles qui nous interesse
    liste_triangles = [[points[i] for i in indices_list]
                       for indices_list in liste_indice_triangles]
    surf.plot(show_edges=True)

    return np.array(liste_triangles)


def add_list_at_each_rows_of_array(array: np.ndarray, list: List) -> np.ndarray:
    """
    Add a list at each row of a 2D array.

    Parameters:
    - array (np.ndarray): The input 2D array.
    - lst (List): The list to be added to each row.

    Returns:
    - np.ndarray: The new array after adding the list to each row.

    Raises:
    - ValueError: If the input array is not a 2D array.
    """
    is_homogenous_array_of_dim(array)
    new_array = []
    array = array.tolist()
    for i in range(len(array)):
        new_array.append(array[i]+list)
    return np.array(new_array)


def array_to_line(array: np.ndarray) -> np.ndarray:
    """
    Convert a 2D or 3D array to a 2D array.

    Parameters:
    - array (np.ndarray): The input array.

    Returns:
    - np.ndarray: The converted 2D array.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    if len(np.shape(array)) == 2:
        return array
    is_homogenous_array_of_dim(array)
    return array.reshape((np.shape(array)[0]*np.shape(array)[1], 3))


def line_to_3Darray(line: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a 1D array to a 3D array with the specified shape.

    Parameters:
    - line (np.ndarray): The input 1D array.
    - shape (Tuple[int, int]): The desired shape of the output 3D array.

    Returns:
    - np.ndarray: The converted 3D array.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    is_homogenous_array_of_dim(line)
    return line.reshape((shape[0], shape[1], 3))


def give_min_mean_max_of_array(array: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate the minimum, mean, and maximum values of a 1D array.

    Parameters:
    - array (np.ndarray): The input 1D array.

    Returns:
    - Tuple[float, float, float]: The minimum, mean, and maximum values of the array.

    Raises:
    - ValueError: If the input array is not a 1D array.
    """
    is_homogenous_array_of_dim(array, 1)

    array_max = max(array)
    array_min = min(array)
    array_mean = sum(array) / len(array)
    return array_min, array_mean, array_max


def convert_rgb_array_to_hsv_array(colors: np.ndarray) -> np.ndarray:
    """
    Convert an array of RGB colors to an array of HSV colors.

    Parameters:
    - colors (np.ndarray): The input array of RGB colors with shape (N, 3).

    Returns:
    - np.ndarray: The converted array of HSV colors.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    is_homogenous_array_of_dim(colors, 3)
    colorshsv = np.asarray([[i, i, i] for i in range(len(colors))])
    for i in range(len(colors)):
        r = colors[i][0]/255
        g = colors[i][1]/255
        b = colors[i][2]/255
        maximum = max([r, g, b])
        minimum = min([r, g, b])
        v = maximum
        if (v == 0):
            s = 0
        else:
            s = (maximum-minimum)/maximum

        if (maximum-minimum == 0):
            h = 0
        else:
            if (v == r):
                h = 60*(g-b)/(maximum-minimum)

            if (v == g):
                h = 120 + 60*(b-r)/(maximum-minimum)

            if (v == b):
                h = 240+60*(r-g)/(maximum-minimum)

        if (h < 0):
            h = h+360

        h = h/360
        colorshsv[i][0] = h*255
        colorshsv[i][1] = s*255
        colorshsv[i][2] = v*255
    return colorshsv


def get_color_3D_array_depending_on_axis(points: np.ndarray, axis: str) -> np.ndarray:
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
    is_homogenous_array_of_dim(points, 3)

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
    min_coord, mean_coord, max_coord = give_min_mean_max_of_array(
        axis_coordinates)

    for i in range(len(colors)):
        colors[i] = pf.convert_float_in_range_to_rgb(
            points[i][axis_int], min_coord, mean_coord, max_coord)
    return np.array(colors)


def centering_3Darray_on_mean_points(array: np.ndarray) -> np.ndarray:
    """
    Center a 3D array of points around their mean.

    Parameters:
    - array (np.ndarray): The input array of 3D points.

    Returns:
    - np.ndarray: The array of 3D points centered around their mean.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    is_homogenous_array_of_dim(array, 3)
    tab_x = []
    tab_y = []
    tab_z = []

    for i in array:
        tab_x.append(i[0])
        tab_y.append(i[1])
        tab_z.append(i[2])

    milieu_x = np.mean(tab_x)
    milieu_y = np.mean(tab_y)
    milieu_z = np.mean(tab_z)

    pt_milieu = [milieu_x, milieu_y, milieu_z]

    new_vertices = [point - pt_milieu for point in array]

    return np.array(new_vertices)

# Arrays and points clouds


def remove_points_of_array_below_threshold(points: np.ndarray, threshold: float, colors: np.ndarray = [], axis: str = "z") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove points from a 3D array based on a threshold along a specific axis.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - threshold (float): The threshold value for removing points.
    - colors (np.ndarray, optional): The array of colors associated with the points.
    - axis (str, optional): The axis along which to apply the threshold ('x', 'y', or 'z').

    Returns:
    - np.ndarray: The filtered array of 3D points (and colors if provided).

    Raises:
    - ValueError: If the input array is not of the correct shape or if the axis is unknown.
    """
    is_homogenous_array_of_dim(points, 3)

    if axis != "x" and axis != "y" and axis != "z":
        raise ValueError(f"Unknown axis: {axis}. Must be x, y or z")

    if axis == "x":
        axis_int = 0
    if axis == "y":
        axis_int = 1
    if axis == "z":
        axis_int = 2

    # Select indices of points whose coordinate along the specified axis is greater than or equal to the threshold
    index = np.where(points[:, axis_int] >= threshold)

    # Check if the filtered point cloud is not empty
    if len(index[0]) == 0:
        raise ValueError("Every points have been removed")

    if len(colors) > 0:
        new_points = points[index]
        new_colors = colors[index]
        return np.array(new_points), np.array(new_colors)
    else:
        new_points = points[index]
        return np.array(new_points)


def reduce_density_of_array(points: np.ndarray, density: float, colors: np.ndarray = []) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reduce the density of a 3D point cloud by randomly selecting a fraction of the points.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - density (float): The fraction of points to keep in the range (0, 1).
    - colors (np.ndarray, optional): The array of colors associated with the points.

    Returns:
    - np.ndarray: The reduced array of 3D points (and colors if provided).

    Raises:
    - ValueError: If the input array is not of the correct shape or if the density is outside the valid range.
    """
    is_homogenous_array_of_dim(points, 3)

    if not (0 < density <= 1):
        raise ValueError("Density must be in the range (0, 1]")

    colors_reduits = []

    # Calculate the number of points to keep
    nombre_points = int(len(points) * density)

    # Randomly select indices of points to keep
    indices_a_conserver = np.random.choice(
        len(points), nombre_points, replace=False)

    # Extract the selected points
    points_reduits = points[indices_a_conserver, :]
    if len(colors) > 0:
        colors_reduits = colors[indices_a_conserver, :]
        return np.array(points_reduits), np.array(colors_reduits)

    return np.array(points_reduits)


def filter_array_with_sphere_on_barycentre(points: np.ndarray, radius: float, colors: np.ndarray = [], tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter a 3D point cloud by keeping only the points within a sphere centered at the barycentre.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - radius (float): The radius of the sphere.
    - colors (np.ndarray, optional): The array of colors associated with the points.
    - indices (List[int], optional): The array of indices associated with the points.

    Returns:
    - np.ndarray: The filtered array of 3D points (and colors and indices if provided).

    Raises:
    - ValueError: If the input array is not of the correct shape or if the radius is negative.
    """
    is_homogenous_array_of_dim(points, 3)
    if radius < 0:
        raise ValueError("Radius must be non-negative")

    barycentre = np.mean(points, axis=0)
    rayon_filtrage = radius
    filtered_points = []
    filtered_colors = []
    filtered_indice = []

    for i, point in enumerate(points):
        distance = np.linalg.norm(point - barycentre)
        if distance <= rayon_filtrage:
            filtered_points.append(point)
            if len(colors) > 0:
                filtered_colors.append(colors[i])
            if len(tableau_indice) > 0:
                filtered_indice.append(tableau_indice[i])
    if len(tableau_indice) > 0:
        return np.array(filtered_points), np.array(filtered_colors), np.array(filtered_indice)
    if len(colors) > 0:
        return np.array(filtered_points), np.array(filtered_colors)
    return np.array(filtered_points)

# Points clouds


def crop_pc_from_zone_selection(points: np.ndarray, colors: np.ndarray, shape: Tuple[int, int] = [], tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Crop a 3D point cloud based on user-defined rectangle selection.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - colors (np.ndarray): The array of colors associated with the points.
    - shape (Tuple[int, int], optional): The shape of the display window. Default is an empty tuple.
    - indices (List[int], optional): The array of indices associated with the points.

    Returns:
    - Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: The cropped array of 3D points, colors, and indices.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    # Globa variables to store information of mouseclick
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
            cv2.imshow("Cropping", colors_image)

    if shape != []:
        if np.shape(shape) != (2,):
            raise ValueError(f"Incorrect shape {shape} for the display")
        length, height = shape
        colors_image = line_to_3Darray(
            colors, (height, length)).astype(np.uint8)
    else:
        if len(np.shape(colors)) != 3:
            raise ValueError(
                f"Incorrect shape for the display got {np.shape(colors)} and expected (x,y,z)")
        if np.shape(colors)[2] != 3:
            raise ValueError(
                f"Incorrect dimension for the array, expected 3 and given {np.shape(colors)[2]}")
        colors_image = colors.astype(np.uint8)
        colors = array_to_line(colors)
        height, length = np.shape(colors_image)[0], np.shape(colors_image)[1]

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
    bottom_left_corner = (y_min-1)*length + x_min
    top_left_corner = (y_max-1)*length + x_min
    bottom_right_corner = (y_min-1)*length + x_max

    i = 0
    points_cloud_crop = []
    couleurs_crop = []
    tableau_indice_crop = []

    while (bottom_left_corner != top_left_corner):
        for j in range(bottom_left_corner, bottom_right_corner):
            points_cloud_crop.append(points[j])
            couleurs_crop.append(colors[j])
            if len(tableau_indice) > 0:
                tableau_indice_crop.append(tableau_indice[j])
        bottom_left_corner = (y_min+i-1)*length + x_min
        bottom_right_corner = (y_min+i-1)*length + x_max
        i += 1

    cv2.destroyAllWindows()
    
    if len(tableau_indice_crop) > 0:
        return np.array(points_cloud_crop), np.array(couleurs_crop), np.array(tableau_indice_crop)
    else:
        return np.array(points_cloud_crop), np.array(couleurs_crop)


def apply_hsv_mask_to_pc(points: np.ndarray, colors: np.ndarray, maskhsv: Tuple[np.ndarray, np.ndarray], tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Apply an HSV mask to a 3D point cloud.

    Parameters:
    - points (np.ndarray): The input array of 3D points.
    - colors (np.ndarray): The array of colors associated with the points.
    - mask_hsv (Tuple[np.ndarray,np.ndarray]): The HSV mask to apply. The mask must be in the following from : [[H_L,S_L,V_L],[H_U,S_U,V_U]] with 
    H_L,S_L,V_L the lower values of Hue, Saturation and Value, and H_L,S_L,V_L the upper values. 
    You can generate such mask using the function "get_hsv_mask_with_sliders" in "processing_img.py".
    - indices (List[int], optional): The array of indices associated with the points.

    Returns:
    - Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: The filtered array of 3D points, colors, and indices (if provided).

    Raises:
    - ValueError: If the input arrays are not of the correct shape.
    """
    # Convert RGB colors to HSV
    colorshsv = convert_rgb_array_to_hsv_array(colors)
    # Construct the mask
    msk = [False for i in range(0, len(colors))]
    for i in range(0, len(colors)):
        # Condition : the three hsv values of the colors must be in maskhsv
        if ((colorshsv[i][0] > maskhsv[0][0]) and (colorshsv[i][0] < maskhsv[1][0])):  # Composante h
            # s
            if ((colorshsv[i][1] > maskhsv[0][1]) and (colorshsv[i][1] < maskhsv[1][1])):
                # v
                if ((colorshsv[i][2] > maskhsv[0][2]) and (colorshsv[i][2] < maskhsv[1][2])):
                    msk[i] = True

    # Apply the mask to filter points, colors, and indices (if provided)
    points = points[msk]
    colors = colors[msk]

    if len(tableau_indice) > 0:
        tableau_indice = tableau_indice[msk]
        return np.array(points), np.array(colors), np.array(tableau_indice)

    return np.array(points), np.array(colors)


def center_pc_on_image(points: np.ndarray, colors: np.ndarray, image_target: str, shape: Tuple[int, int] = []) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center a point cloud on the target image.

    Parameters:
    - points (np.ndarray): The array of 3D points.
    - colors (np.ndarray): The array of colors associated with the points.
    - image_target (str): The path to the target image.
    - shape (Tuple[int, int], optional): The shape of the image.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The centered array of 3D points and colors.
    """
    # Get the size of the target image

    largeur, hauteur = pi.get_size_of_image(image_target)

    # Calculate the center of the image in pixel coordinates
    centre_image_ref = np.array(
        [int(largeur/2), int(hauteur/2), 1])  # En pixel

    image_source = "center_pc_on_image.png"

    # Save the colors as an image for homography calculation
    pi.save_image_from_array(colors, image_source, shape)

    # Get the size of the source image
    longueur, _ = pi.get_size_of_image(image_source)

    # Calculate the coordinates of the center in the source image (in pixel coordinates)
    homography_matrix = pi.get_homography_between_imgs(
        image_target, image_source)
    pg.delete_file(image_source)

    projected_center = np.dot(homography_matrix, centre_image_ref)  # en pixel

    # Convert to depth coordinates
    origine_nuage = points[(int(projected_center[1])-1)
                           * longueur + int(projected_center[0])]

    # Reposition the point cloud to be centered around the origin point
    nuage_point_centred = [(x[0] - origine_nuage[0], x[1] -
                            origine_nuage[1], x[2] - origine_nuage[2]) for x in points]

    return nuage_point_centred, colors


if __name__ == '__main__':
    # l = np.array([1, 2, 3])
    points, colors = pp.get_points_and_colors_of_ply("./example/test.ply")
    # crop_pc_from_zone_selection(points, colors, (640, 480))
    new_colors = line_to_3Darray(colors, (480, 640))
    # pi.save_image_from_array(new_colors,"oui.png")
    # plot_3D_array(points)
    # print(add_list_at_each_rows_of_array(points, [0., 0., 0., 1.])[0])
    print(array_to_line(colors))
    # test=[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]
    # liste_triangles=build_mesh_from_3Darray(test)
