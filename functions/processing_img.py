import numpy as np
import logging
import cv2
import sys

from typing import Tuple, Optional


mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from .utils import array as array
else:
    # Code executed as a script
    import utils.array as array

def load(image_path:str,mode:int=-1):
    """
    LOAD GRAYSCALE => mode = 0
    """
    image = cv2.imread(image_path,mode)
    if image is None:
        raise ValueError(
            f"The image is empty. Please check if the path {image_path} is correct ")
    # Load the image
    return image

def save(pixels: np.ndarray, output_filename: str, shape: Optional[Tuple[int, int]] = []) -> None:
    """
    Save an image from a pixel array to a file.

    Parameters:
    - pixels: The pixel array (2D or 3D).
    - output_filename (str): The output file name.
    - shape (Tuple[int, int], optional): The shape of the image for 2D arrays.

    Raises:
    - ValueError: If the shape is incorrect for display or if the array dimension is not supported.

    Returns:
    - None: The function does not return anything, but it saves an image file.
    """
    if shape == []:
        cv2.imwrite(output_filename, pixels)
        logging.info(f"Image saved under the name '{output_filename}'.")
    else:
        if np.shape(shape) != (2,):
            raise ValueError(f"Incorrect shape {shape} for the display")
        pixels = array.line_to_3Darray(pixels, (shape[0], shape[1]))
        cv2.imwrite(output_filename, pixels)
        logging.info(f"Image saved under the name '{output_filename}'.")


# image1, image2 = image.load_image("img1"), image.load_image("img1")

# image1 = pnglib.load("png")
# image2 = jpglib.load("jpg")

# homography = get_homography(image1, image2)
# image3= apply_homograpy(image2, homgography)

# affiche( image3 )


if __name__ == '__main__':
    # print(get_shining_point_image("./example/image_source.png"))
    # array=give_array_from_image("./example/image_source.png")
    # save_image_from_array(array,"./example/image_source_rebuilt.png",(640,480))
    pixels=load("./example/capture_realsense.png",0)
    save(pixels,"test.png")
    # H = get_homography_between_imgs("./example/image_ref.png", "./example/image_source.png", True)
    # print(H)
