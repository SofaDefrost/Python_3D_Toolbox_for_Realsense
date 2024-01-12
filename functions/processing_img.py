import numpy as np
import logging
import cv2
import sys

from typing import Tuple, Optional


mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from .utils import array as array
    from . import processing_pixel_list as pixels
else:
    # Code executed as a script
    import utils.array as array
    import processing_pixel_list as pixels


def load(image_path: str, mode: Optional[int] = 1) -> Optional[np.ndarray]:
    """
    Load an image from a file.

    Parameters:
        image_path (str): The path to the image file.
        mode (int, optional): The mode to read the image (default is 1, which loads the image as it is).

    Returns:
        np.ndarray: The loaded image.

    Raises:
        ValueError: If the image is empty or the provided path is incorrect.
    """
    image = cv2.imread(image_path, mode)
    if image is None:
        raise ValueError(
            f"The image is empty. Please check if the path {image_path} is correct ")
    # Load the image
    return image[:, :, ::-1]


def save(pixels: np.ndarray, output_filename: str, shape: Optional[Tuple[int, int]] = []) -> None:
    """
    Save an image from a pixel array to a file.

    Parameters:
    - pixels: The pixel array.
    - output_filename (str): The output file name.
    - shape (Tuple[int, int], optional): The shape of the image for 2D arrays.

    Raises:
    - ValueError: If the shape is incorrect for display or if the array dimension is not supported.

    Returns:
    - None: The function does not return anything, but it saves an image file.
    """
    if shape == []:
        cv2.imwrite(output_filename, pixels[:, :, ::-1])
        logging.info(f"Image saved under the name '{output_filename}'.")
    else:
        if np.shape(shape) != (2,):
            raise ValueError(f"Incorrect shape {shape} for the display")
        pixels = array.line_to_3Darray(pixels, (shape[0], shape[1]))
        cv2.imwrite(output_filename, pixels[:, :, ::-1])
        logging.info(f"Image saved under the name '{output_filename}'.")


if __name__ == '__main__':
    # Loading colors
    image1 = load("./example/input/image_ref.png")
    image2 = load("./example/input/image_source.png")
    # Test homography
    H = pixels.get_homography(image1, image2)
    transformed_image1 = pixels.apply_transformation_matrix(image1, H)
    image2_with_homography = pixels.add_polygon(
        image2, transformed_image1, (0, 255, 0))
    pixels.display(image2_with_homography, "Homography test")
    # Test shining point detection
    shining_point = pixels.get_shining_point(image2)
    image2_with_shining_point = pixels.add_point(
        image2, shining_point[0], shining_point[1], (0, 255, 0))
    pixels.display(image2_with_shining_point, "Shining point detection test")
    # HSV Mask
    Mask = pixels.get_hsv_mask_with_sliders(image2)
    shining_point_hsv = pixels.get_shining_point_with_hsv_mask(image2, Mask)
    image2_with_shining_point_hsv = pixels.add_point(
        image2, shining_point_hsv[0], shining_point_hsv[1], (0, 255, 0))
    pixels.display(image2_with_shining_point_hsv,
            "Shining point detection test with hsv mask")
    save(image2_with_shining_point_hsv,"./example/output/Shining_point_detection_test_with_hsv_mask.png")
