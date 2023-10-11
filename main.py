import acquisition as aq
import filtre_hsv_realsense as filtre

print("Une petite démo:\nUne première version :")

point_cloud_name = "demo.ply"
color_image_name = "demo.png"

aq.run_acquisition(point_cloud_name, color_image_name)

print("Fichier .ply et .png exportés !\nUne deuxième version :\nPoints et couleurs capturés :")

print(aq.points_and_colors_realsense()[0])
print(aq.points_and_colors_realsense()[1])

print("Dernière démo:\nMasque hsv:")
print(filtre.determinemaskhsv())