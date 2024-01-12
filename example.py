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

from functions.utils import array as array
from functions import processing_ply as ply
from functions import processing_img as img
from functions import processing_pixel_list as pixels
from functions import apply_functions_display as aTk
from functions import processing_point_cloud as pc

# Init acquisition
pipeline=aq.init_realsense(640,480)

points,colors=aq.get_points_and_colors_from_realsense(pipeline)

point_cropped, color_cropped,_ = pc.crop_from_zone_selection(points,colors)

maskhsv = pixels.get_hsv_mask_with_sliders(colors)

points_hsv,colors_hsv,_ = pc.apply_hsv_mask(points,array.to_line(colors),maskhsv)

ply.save("example/output/capture_realsense_hsv.ply",points_hsv,colors_hsv)

# pp.filter_array_with_sphere_on_barycentre_with_interface(
#     input_filename="example/masked_capture_realsense.ply", output_filename="example/sphere_capture_realsense.ply")

# pp.color_ply_depending_on_axis(input_filename="example/capture_realsense.ply",
#                                output_filename="example/capture_realsense_rainbow_colored.ply", axis="x")

# pp.centering_ply_on_mean_points(input_filename="example/capture_realsense.ply",
#                                 output_filename="example/capture_realsense_centered.ply")

# pp.center_ply_on_image(input_filename="example/capture_with_image_ref.ply",
#                        output_filename="example/capture_centered_on_image_ref.ply", image_target_path="example/image_ref.png", shape=(640, 480))


print("All outcomes will be stored in the 'example/output' folder.")