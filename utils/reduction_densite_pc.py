import numpy as np
import open3d as o3d


# # Cette fonction réduit le nombre de point d'un .ply selon une certaine densité (par exemple densite=0.5 signifie que vous conservez 50% des points)
def reduction_densite_pc(input_file, output_file, densite):
    # Charger le nuage de points
    cloud = o3d.io.read_point_cloud(input_file)

    # Convertir le nuage de points en un tableau NumPy
    points = np.asarray(cloud.points)

    # Convertir les couleurs du nuage de points en un tableau NumPy
    colors = np.asarray(cloud.colors)

    # Calculer le nombre de points à conserver
    nombre_points = int(len(points) * densite)

    # Sélectionner aléatoirement les indices des points à conserver
    indices_a_conserver = np.random.choice(len(points), nombre_points, replace=False)

    # Extraire les points sélectionnés
    points_reduits = points[indices_a_conserver, :]
    if len(colors)!=0:
        colors_reduits = colors[indices_a_conserver, :]

    # Créer un nouveau nuage de points avec les points réduits et les couleurs correspondantes
    cloud_reduit = o3d.geometry.PointCloud()
    cloud_reduit.points = o3d.utility.Vector3dVector(points_reduits)
    if len(colors)!=0:
        cloud_reduit.colors = o3d.utility.Vector3dVector(colors_reduits)

    # Sauvegarder le nuage de points réduit
    o3d.io.write_point_cloud(output_file, cloud_reduit)

    # print(f"Nuage de points réduit enregistré sous : {output_file}")
