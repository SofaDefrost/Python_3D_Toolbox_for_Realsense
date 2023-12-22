"""
    This document includes various examples for handling PLY files. 
    When the code is executed, it initiates the acquisition of the RealSense Camera 
    and processes the obtained file in the following steps:
        - The user can select a specific area of the point cloud acquired by drawing a rectangle on an image.        -
        - The PLY file acquired will be colored based on its axes.
        - The PLY file acquired will be repositioned so that its mean point becomes the center.
        - Finally, another PLY file (named capture_with_image_ref.ply) will be repositioned to the center of the "image_ref.png" picture.
    All outcomes will be stored in the 'example' folder.
"""

import acquisition_realsense as aq

from utils import processing_ply as pp
from utils import processing_img as pi
from utils import display_function_Tkinter as dpt
from utils import processing_array as pa


aq.save_ply_from_realsense_with_interface(
    path_name_ply="example/capture_realsense.ply", image_name="example/capture_realsense.png")

pp.crop_ply_from_pixels_selection(input_filename="example/capture_realsense.ply",
                                  output_filename="example/capture_cropped.ply", shape=(640, 480))

maskhsv = pi.get_hsv_mask_with_sliders(
    image_path="example/capture_realsense.png")

pp.apply_hsv_mask_to_ply(input_filename="example/capture_cropped.ply",
                         output_filename="example/masked_capture_realsense.ply", maskhsv=maskhsv)

pp.filter_array_with_sphere_on_barycentre_with_interface(
    input_filename="example/masked_capture_realsense.ply", output_filename="example/sphere_capture_realsense.ply")

pp.color_ply_depending_on_axis(input_filename="example/capture_realsense.ply",
                               output_filename="example/capture_realsense_rainbow_colored.ply", axis="x")

pp.centering_ply_on_mean_points(input_filename="example/capture_realsense.ply",
                                output_filename="example/capture_realsense_centered.ply")

pp.center_ply_on_image(input_filename="example/capture_with_image_ref.ply",
                       output_filename="example/capture_centered_on_image_ref.ply", image_target_path="example/image_ref.png", shape=(640, 480))


print("All outcomes will be stored in the 'example' folder.")
