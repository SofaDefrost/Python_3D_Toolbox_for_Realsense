import numpy as np

import acquisition_realsense as aq

from utils import processing_ply as pp
from utils import processing_img as pi

aq.save_ply_from_realsense("test_realsense.ply")
pp.crop_ply_from_pixels_selection("test_realsense.ply","test_realsense_cropped.ply",(640,480))
pp.color_ply_depending_on_axis("test_realsense.ply","test_realsense_colored.ply","x")
pp.centering_ply_on_mean_points("test_realsense.ply","test_realsense_centered.ply")
pi.get_homography_between_imgs("image_ref.png","image_source.png",True)