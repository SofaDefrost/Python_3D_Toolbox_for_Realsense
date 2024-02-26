import numpy as np
import logging
import cv2
import sys

from typing import List, Tuple

mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from .utils import array as array
    from . import processing_point_cloud as pc
else:
    # Code executed as a script
    import utils.array as array
    import processing_point_cloud as pc

def add_point(pixels: np.ndarray, coordinate_x: int, coordinate_y: int, colors_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Adds a colored point to an image by first copying the existing pixels.

    Parameters:
        pixels (np.ndarray): Image to add the point to.
        coordinate_x (int): X-coordinate of the point to add.
        coordinate_y (int): Y-coordinate of the point to add.
        colors_rgb (Tuple[int, int, int]): RGB color of the point to add.

    Returns:
        np.ndarray: Resulting image with the added point.
    """
    pixels_copy = np.array(pixels)
    return cv2.circle(pixels_copy, (coordinate_x, coordinate_y), 5, colors_rgb, -1)


def add_polygon(pixels: np.ndarray, polygon_vertex: np.ndarray, color_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Adds a polygon with the specified vertices to an image by first copying the existing pixels.

    Parameters:
        pixels (np.ndarray): Image to add the polygon to.
        polygon_vertex (np.ndarray): Vertices of the polygon to add.
        color_rgb (Tuple[int, int, int]): RGB color of the polygon.

    Returns:
        np.ndarray: Resulting image with the added polygon.
    """
    # Draw the contours of the reference image on the source image
    pixels_copy = np.array([i for i in pixels])
    return cv2.polylines(pixels_copy, [np.int32(polygon_vertex)], True, color_rgb, 2)


def apply_transformation_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Applies a transformation matrix to the corners of the input image.

    Parameters:
        image (np.ndarray): Input image to be transformed.
        matrix (np.ndarray): Transformation matrix.

    Returns:
        np.ndarray: Transformed image.
    """
    # Get the height and width of the image
    h, w = image.shape[:2]
    # Define the corners of the image
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    # Apply perspective transformation to the corners
    return cv2.perspectiveTransform(pts, matrix)


def display(pixels: np.ndarray, window_name: str, shape: List[int] = []) -> None:
    """
    Displays the image using OpenCV's imshow function.

    Parameters:
        pixels (np.ndarray): Image data.
        window_name (str): Name of the window.
        shape (List[int], optional): Shape of the image (height, width) if applicable. Defaults to [].
    """
    if shape != []:
        pixels = array.line_to_2Darray(pixels, (shape[1], shape[0]))
    pixels = pixels.astype(np.uint8) 
    # Display the resulting image
    cv2.imshow(window_name, pixels[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_homography(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Find the homography matrix between two images using ORB features.

    Parameters:
        image1 (np.ndarray): The first image.
        image2 (np.ndarray): The second image.

    Returns:
        np.ndarray: The 3x3 homography matrix.
    """
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    image1_grayscale = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_grayscale = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1_grayscale, None)
    kp2, des2 = orb.detectAndCompute(image2_grayscale, None)

    # Use the BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on their similarity
    matches = sorted(matches, key=lambda x: x.distance)

    # Select the best matches (can adjust the ratio accordingly)
    good_matches = matches[:50]

    # Get corresponding points in both images
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H


def get_hsv_mask_with_sliders(image: np.ndarray,shape:List[int]= []) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtain an HSV mask using sliders.

    Args:
        image (np.ndarray): Input image in RGB format.
        shape (List[int], optional): Shape of the image (height, width) if applicable. Defaults to [].

    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper HSV values.
    """
    
    if shape != []:
        image = array.line_to_2Darray(image,(shape[1],shape[0]))
    
    image = cv2.convertScaleAbs(image)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)

    def update_mask_hsv(x):
        global lower_hsv, upper_hsv, mask, result
        lower_hsv = np.array([cv2.getTrackbarPos('Hue Min', 'HSV Interface'),
                              cv2.getTrackbarPos('Satur Min', 'HSV Interface'),
                              cv2.getTrackbarPos('Value Min', 'HSV Interface')])

        upper_hsv = np.array([cv2.getTrackbarPos('Hue Max', 'HSV Interface'),
                              cv2.getTrackbarPos('Satur Max', 'HSV Interface'),
                              cv2.getTrackbarPos('Value Max', 'HSV Interface')])

        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
        cv2.imshow('HSV Mask', mask)
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('Result', result)
        return lower_hsv, upper_hsv
    image = image[:, :, ::-1]  # Conversion RGB to BGR
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create windows
    cv2.namedWindow('Original Image')
    cv2.namedWindow('HSV Interface')
    cv2.namedWindow('HSV Mask')
    cv2.namedWindow('Result')

    # Create sliders for HSV components
    cv2.createTrackbar('Hue Min', 'HSV Interface', 0, 179, update_mask_hsv)
    cv2.createTrackbar('Hue Max', 'HSV Interface', 179, 179, update_mask_hsv)
    cv2.createTrackbar('Satur Min', 'HSV Interface', 0, 255, update_mask_hsv)
    cv2.createTrackbar('Satur Max', 'HSV Interface', 255, 255, update_mask_hsv)
    cv2.createTrackbar('Value Min', 'HSV Interface', 0, 255, update_mask_hsv)
    cv2.createTrackbar('Value Max', 'HSV Interface', 255, 255, update_mask_hsv)

    # Initialize HSV ranges
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([179, 255, 255])

    # Create an initial black image as the initial mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    result = image.copy()

    # Display the original image
    cv2.imshow('Original Image', image)

    print("Press the 'q' key to export the mask.")

    while True:
        # Update the mask based on the sliders
        lower_hsv, upper_hsv = update_mask_hsv(0)

        # Display the three windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Once the user presses 'q', return the mask
    cv2.destroyAllWindows()
    logging.info(f"Mask hsv exported !")

    return lower_hsv, upper_hsv


def get_shining_point(pixels: np.ndarray) -> Tuple[int, int]:
    """
    Detect the brightest point in an RGB image.

    Parameters:
        pixels (np.ndarray): The input RGB image.

    Returns:
        Tuple[int, int]: Coordinates (x, y) of the brightest point.
    """
    # Convert the array to an OpenCV image
    image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the cornerHarris function to detect the brightest corner
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Normalize the response to make it easier to detect the brightest point
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)

    # Find the coordinates of the shining point
    y, x = np.unravel_index(dst.argmax(), dst.shape)

    return x, y


def get_shining_point_with_hsv_mask(image: np.ndarray, hsv_mask: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, int]:
    """
    Detect the brightest point within a specified HSV color mask.

    Parameters:
        image (np.ndarray): The input BGR image.
        hsv_mask (Tuple[np.ndarray, np.ndarray]): The HSV color range for the mask.

    Returns:
        Tuple[int, int]: Coordinates (x, y) of the brightest point within the mask.
    """

    lower_red = hsv_mask[0]
    upper_red = hsv_mask[1]

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask using the defined color range
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_sum = 0
    y_sum = 0
    sum = 0
    # Iterate over all detected contours
    for contour in contours:
        # Get the coordinates of pixels in the contour
        for point in contour:
            x, y = point[0]
            x_sum += x
            y_sum += y
            sum += 1
    pixel_x = int(x_sum/sum)
    pixel_y = int(y_sum/sum)

    return pixel_x, pixel_y


def convert_from_rgb_to_hsv(pixels: np.ndarray) -> np.ndarray:
    """
    Convert an array of RGB colors to an array of HSV colors.

    Parameters:
    - pixels (np.ndarray): The input array of RGB colors with shape (N, 3).

    Returns:
    - np.ndarray: The converted array of HSV colors.

    Raises:
    - ValueError: If the input array is not of the correct shape.
    """
    array.is_homogenous_of_dim(pixels, 3)

    r, g, b = pixels[:, 0] / 255, pixels[:, 1] / 255, pixels[:, 2] / 255
    maximum = np.maximum.reduce([r, g, b])
    minimum = np.minimum.reduce([r, g, b])
    v = maximum

    # Compute saturation
    s = np.where(v == 0, 0, (maximum - minimum) / maximum)

    # Compute hue
    h = np.zeros_like(v)
    non_zero_mask = (maximum - minimum) != 0

    h[non_zero_mask] = np.where(v == r[non_zero_mask], 60 * (g[non_zero_mask] - b[non_zero_mask]) / (maximum[non_zero_mask] - minimum[non_zero_mask]),
                                np.where(v == g[non_zero_mask], 120 + 60 * (b[non_zero_mask] - r[non_zero_mask]) / (maximum[non_zero_mask] - minimum[non_zero_mask]),
                                         240 + 60 * (r[non_zero_mask] - g[non_zero_mask]) / (maximum[non_zero_mask] - minimum[non_zero_mask])))

    h[h < 0] += 360
    h /= 360

    # Convert to 8-bit integer values
    colorshsv = np.column_stack((h * 255, s * 255, v * 255))

    return colorshsv
