import numpy as np
import Sofa.Core

from Python_3D_Toolbox_for_Realsense import acquisition_realsense as aq
from Python_3D_Toolbox_for_Realsense.functions import processing_point_cloud as pc
from Python_3D_Toolbox_for_Realsense.functions import processing_pixel_list as pixels

class realsense_maj_points(Sofa.Core.Controller):

        def __init__(self,node,name,pipeline,mask_hsv,size_acqui,*args, **kwargs):
            Sofa.Core.Controller.__init__(self,args,kwargs)
            self.stiffNode = node # for the generic one
            self.position = self.stiffNode.getObject(name)
            self.pipeline=pipeline
            self.mask_hsv = mask_hsv
            self.size_acqui=size_acqui

        def onAnimateBeginEvent(self,e):
            points,colors = aq.get_points_and_colors_from_realsense(self.pipeline)
            points_filtered_hsv, _, _ = pc.apply_hsv_mask(points, colors, self.mask_hsv, self.size_acqui)
            nb_point_tot = self.size_acqui[0]*self.size_acqui[1]
            
            missing_elements = nb_point_tot - points_filtered_hsv.shape[0]
    
            # Créer un tableau de zéros de la taille nécessaire
            zeros = np.zeros((missing_elements, 3))

            # Concaténer le tableau original avec les zéros
            completed_array = np.concatenate((points_filtered_hsv, zeros))
            
            self.position.position.value = completed_array


        def onAnimateEndEvent(self, event):
            # Delete every points
            self.reset()