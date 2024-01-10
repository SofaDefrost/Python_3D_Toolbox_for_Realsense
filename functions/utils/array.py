import numpy as np

from typing import List, Optional, Tuple


def is_homogenous_of_dim(array: np.ndarray, dimension: Optional[int] = -1) -> None:
    """
    Check if the input array is homogeneous and has the specified dimension.

    Parameters:
    - array (List[Union[int, float, str]]): The input array to be checked.
    - dimension (int, optional): The expected dimension of the array. If not specified, any dimension is allowed.

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


def add_list_at_each_rows(array: np.ndarray, list: List) -> np.ndarray:
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
    is_homogenous_of_dim(array)
    new_array = []
    array = array.tolist()
    for i in range(len(array)):
        new_array.append(array[i]+list)
    return np.array(new_array)


def to_line(array: np.ndarray) -> np.ndarray:
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
    is_homogenous_of_dim(array)
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
    is_homogenous_of_dim(line)
    return line.reshape((shape[0], shape[1], 3))


def give_min_mean_max(array: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate the minimum, mean, and maximum values of a 1D array.

    Parameters:
    - array (np.ndarray): The input 1D array.

    Returns:
    - Tuple[float, float, float]: The minimum, mean, and maximum values of the array.

    Raises:
    - ValueError: If the input array is not a 1D array.
    """
    is_homogenous_of_dim(array, 1)

    array_max = max(array)
    array_min = min(array)
    array_mean = sum(array) / len(array)
    return array_min, array_mean, array_max


def convert_from_rgb_to_hsv(colors: np.ndarray) -> np.ndarray:
    """
    Convert an array of RGB colors to an array of HSV colors.

    Parameters:
    - colors (np.ndarray): The input array of RGB colors with shape (N, 3).

    Returns:
    - np.ndarray: The converted array of HSV colors.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    is_homogenous_of_dim(colors, 3)
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


if __name__ == '__main__':
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    print(add_list_at_each_rows(points, [0., 0., 0., 1.]))
