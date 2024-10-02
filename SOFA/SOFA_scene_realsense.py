import SOFA.PythonController as controller

from Python_3D_Toolbox_for_Realsense import acquisition_realsense as aq
from Python_3D_Toolbox_for_Realsense.functions import processing_point_cloud as pc
from Python_3D_Toolbox_for_Realsense.functions import processing_pixel_list as pixels

def createScene(rootNode):

    rootNode.addObject('VisualStyle', displayFlags="showCollision" )

    # Creating SOFA object that represent the point cloud

    point_cloud = rootNode.addChild('point_cloud')

    point_cloud.addObject('MechanicalObject', name='MeasuredPositionM0', position=[0, 0,0])
    point_cloud.addObject('SphereCollisionModel', radius='0.0005')#, group='1')
    mechanical_object = point_cloud.getObject("MeasuredPositionM0")
    mechanical_object.init()
    
    # Get points from realsense
    size_acqui = (1280,720)
    pipeline = aq.init_realsense(size_acqui[0],size_acqui[1])
    points,colors = aq.get_points_and_colors_from_realsense(pipeline)
    
    # Get mask

    mask_hsv = pixels.get_hsv_mask_with_sliders(colors, size_acqui)

    # Apply mask

    points_filtered_hsv, _, _ = pc.apply_hsv_mask(points, colors, mask_hsv, size_acqui)
    
    # Put all points in SOFA
    mechanical_object.position.value = points_filtered_hsv #new_points_sofa

    # Update points position
    rootNode.addObject(controller.realsense_maj_points(node = point_cloud,name = 'MeasuredPositionM0',pipeline=pipeline,size_acqui=size_acqui,mask_hsv=mask_hsv))
           
    return rootNode
