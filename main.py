#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mode d'emploi : L'utilisateur devra cliquer sur la touche 'S' pour enregistrer le nuage de points au format PLY et l'image optique,

et sur la touche 'q' pour arrêter l'acquisition. De plus,
 il devra fournir un nom de fichier pour le fichier PLY ainsi que pour l'image optique."""
  


import acquisition  as aq
point_cloud_name = "pc.ply"
color_image_name = "image.png"
aq.run_acquisition(point_cloud_name, color_image_name)