import numpy as np
import logging
import cv2
import sys

from PIL import Image
from typing import List, Tuple, Optional


mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from . import processing_point_cloud as pa
else:
    # Code executed as a script
    import processing_point_cloud as pa


def is_readable_image(image_path: str) -> None:
    """
    Check if an image is readable using OpenCV.

    Parameters:
    - image_path (str): The path to the image file.

    Raises:
    - ValueError: If the image is empty or cannot be read. Check if the path is correct.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"The image is empty. Please check if the path {image_path} is correct ")


def get_size_of_image(image_path: str) -> Tuple[int, int]:
    """
    Get the size (width, height) of an image using the PIL library.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - Tuple[int, int]: The width and height of the image.
    """
    img_ref = Image.open(image_path)
    return img_ref.size


def save_image_from_array(pixels: np.ndarray, nom_fichier_sortie: str, shape: Optional[Tuple[int, int]] = []) -> None:
    """
    Save an image from a pixel array to a file using the PIL library.

    Parameters:
    - pixels: The pixel array (2D or 3D).
    - nom_fichier_sortie (str): The output file name.
    - shape (Tuple[int, int], optional): The shape of the image for 2D arrays.

    Raises:
    - ValueError: If the shape is incorrect for display or if the array dimension is not supported.

    Returns:
    - None: The function does not return anything, but it saves an image file.
    """
    if shape == []:
        if len(np.shape(pixels)) != 3:
            raise ValueError(
                f"Incorrect shape got {np.shape(pixels)} and expected (x,y,z)")
        if np.shape(pixels)[2] != 3:
            raise ValueError(
                f"Incorrect dimension for the array, expected 3 and given {np.shape(pixels)[2]}")
        image = Image.fromarray(pixels.astype(np.uint8))
    else:
        if np.shape(shape) != (2,):
            raise ValueError(f"Incorrect shape {shape} for the display")
        pa.line_to_3Darray(pixels, (shape[0], shape[1]))
        # Create an image with the specified shape
        image = Image.new("RGB", (shape[0], shape[1]))

        # Fill the image with pixels from the list
        # Convert lists to tuples
        pixel_data = [tuple((int(pixel[0]), int(pixel[1]), int(pixel[2])))
                      for pixel in pixels]
        image.putdata(pixel_data)

    # Save the image
    image.save(nom_fichier_sortie)
    logging.info(f"Image saved under the name '{nom_fichier_sortie}'.")


def give_array_from_image(image_name: str) -> np.ndarray:
    """
    Extract an array of RGB values from an image using the PIL library.

    Parameters:
    - image_name (str): The path to the image file.

    Returns:
    - np.ndarray: An array of RGB values representing the pixels in the image.
    """
    image = Image.open(image_name)
    tableau_image = np.array(image)

    # Get the dimensions of the image
    largeur, hauteur, _ = tableau_image.shape

    # Reshape the array to match the dimensions of the image
    tableau_image = tableau_image.reshape((hauteur, largeur, -1))

    # Extract the RGB components
    liste_pixels_rgb = [tuple(pixel)
                        for ligne in tableau_image for pixel in ligne]

    return np.array(liste_pixels_rgb)

# image1, image2 = image.load_image("img1"), image.load_image("img1")

# image1 = pnglib.load("png")
# image2 = jpglib.load("jpg")

# homography = get_homography(image1, image2)
# image3= apply_homograpy(image2, homgography)

# affiche( image3 )


def get_homography_between_imgs(image1_path: str, image2_path: str, display: Optional[bool] = False) -> np.ndarray:
    """
    Compute the homography matrix between two images using the ORB feature detector.

    Parameters:
    - image1_path (str): The path to the reference image.
    - image2_path (str): The path to the target image.
    - display (bool, optional): Whether to display the images with matched points (default is False).

    Returns:
    - np.ndarray: The homography matrix.

    Raises:
    - ValueError: If either image cannot be read.
    """
    is_readable_image(image1_path)
    is_readable_image(image2_path)
    # Load the images in grayscale
    img1 = cv2.imread(image1_path, 0)  # Reference image in grayscale
    img2 = cv2.imread(image2_path, 0)  # Target image in grayscale

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

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
    if display:
        # Load the images in color
        img1_color = cv2.imread(image1_path)
        img2_color = cv2.imread(image2_path)

        # Apply the homography matrix to transform the corners of the reference image
        h, w = img1_color.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                         [w - 1, 0]]).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, H)

        # Draw the contours of the reference image on the source image
        img2_with_reference = cv2.polylines(
            img2_color, [np.int32(transformed_pts)], True, (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow(f'{image1_path} in {image2_path}', img2_with_reference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return H


def get_hsv_mask_with_sliders(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an HSV mask using sliders to define the lower and upper HSV values interactively.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - Tuple[np.ndarray,np.ndarray]: The lower and upper HSV values.

    Raises:
    - ValueError: If the image cannot be read.
    """
    is_readable_image(image_path)

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
        return np.array(lower_hsv), np.array(upper_hsv)

    # Load your image
    image = cv2.imread(image_path)

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
    logging.info(f"Mask hsv [{lower_hsv},{upper_hsv}] exported !")
    return np.array(lower_hsv), np.array(upper_hsv)


def get_shining_point_image(image_path: np.ndarray, display: Optional[bool] = False) -> Tuple[int, int]:
    """
    Detect the brightest point in an image using the cornerHarris algorithm.

    Parameters:
    - image_path (str): The path to the image file.
    - display (bool, optional): Whether to display the image with the detected shining point (default is False).

    Returns:
    - Tuple[int, int]: The coordinates (x, y) of the brightest point.

    Raises:
    - ValueError: If the image cannot be read.
    """
    is_readable_image(image_path)
    # Load your image
    image_array = cv2.imread(image_path)

    # Convert the array to an OpenCV image
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the cornerHarris function to detect the brightest corner
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Normalize the response to make it easier to detect the brightest point
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)

    # Find the coordinates of the shining point
    y, x = np.unravel_index(dst.argmax(), dst.shape)
    if display:
        # Visualization
        # Dessine un cercle rouge
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        # Show the image with the detected shining point
        cv2.imshow('Shining point', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x, y


def get_shining_point_with_hsv_mask(image_path: str, hsv_mask: np.ndarray, display: Optional[bool] = False) -> Tuple[int, int]:
    """
    Detect the shining point in an image using an HSV color mask.

    Parameters:
    - image_path (str): The path to the image file.
    - hsv_mask (Tuple[Tuple[int], Tuple[int]]): The HSV color mask range.
    - display (bool, optional): Whether to display the image with the detected shining point (default is False).

    Returns:
    - Tuple[int, int]: The coordinates (x, y) of the detected shining point.

    Raises:
    - ValueError: If the image cannot be read.
    """
    is_readable_image(image_path)

    lower_red = hsv_mask[0]
    upper_red = hsv_mask[1]

    # Load the image
    image = cv2.imread(image_path)

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

    if display:
        # Draw contours on the original image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Show the image with contours
        cv2.imshow('Image with laser point contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pixel_x, pixel_y


if __name__ == '__main__':
    # print(get_shining_point_image("./example/image_source.png"))
    # array=give_array_from_image("./example/image_source.png")
    # save_image_from_array(array,"./example/image_source_rebuilt.png",(640,480))
    print(get_hsv_mask_with_sliders("./example/image_ref.png"))
    # H = get_homography_between_imgs("./example/image_ref.png", "./example/image_source.png", True)
    # print(H)
