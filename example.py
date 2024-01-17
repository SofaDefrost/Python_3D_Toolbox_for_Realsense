"""
    This document includes various examples for handling PLY files. 
    When the code is executed, it initiates the acquisition of the RealSense Camera 
    and processes the obtained file in the following steps:
        - The user can select a specific area of the point cloud acquired by drawing a rectangle on an image.        -
        - The user will then be able to choose a HSV mask using an interface. This mask will then be applied.
        - Finally, another interface will be opened to help the user to choose the parameter of a filter. 
    At the end, the final result will be saved.
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
# Acquisition
points,colors=aq.get_points_and_colors_from_realsense(pipeline)
# Zone Selection
point_cropped, color_cropped,_,new_shape = pc.crop_from_zone_selection(points,colors)
# Choose mask
maskhsv = pixels.get_hsv_mask_with_sliders(array.line_to_3Darray(color_cropped,new_shape))
# Apply mask
points_hsv,colors_hsv,_ = pc.apply_hsv_mask(point_cropped,color_cropped,maskhsv,new_shape)
# Choose radius filter
radius=aTk.get_parameter_using_preview(points_hsv,pc.filter_with_sphere_on_barycentre,"Radius")
# Apply radius filter
points_filtre,colors_filtre,_=pc.filter_with_sphere_on_barycentre(points_hsv,radius,colors_hsv)
# Save result
ply.save("example/output/capture_realsense_out.ply",points_filtre,colors_filtre)

print("Outcome(s) will be stored in the 'example/output' folder.")