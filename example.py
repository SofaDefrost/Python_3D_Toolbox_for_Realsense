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
from functions import previsualisation_application_function as aTk
from functions import processing_point_cloud as pc

# Init acquisition
pipeline=aq.init_realsense(640,480)
#Acquisition
points,colors=aq.get_points_and_colors_from_realsense(pipeline)
# Zone Selection
point_cropped, color_cropped,_ = pc.crop_from_zone_selection(points,colors)
# Choose mask
maskhsv = pixels.get_hsv_mask_with_sliders(colors)
# Apply mask
points_hsv,colors_hsv,_ = pc.apply_hsv_mask(point_cropped,array.to_line(color_cropped),maskhsv)
# Choose radius filter
radius=aTk.get_parameter_using_preview(point_cropped,pc.filter_with_sphere_on_barycentre,"Radius")
# Apply radius filter
points_filtre,colors_filtre,_=pc.filter_with_sphere_on_barycentre(points_hsv,radius,colors_hsv)
# Save result
ply.save("example/output/capture_realsense_out.ply",points_filtre,colors_filtre)

print("All outcomes will be stored in the 'example/output' folder.")