import open3d as o3d
import numpy as np

# Fonction qui permet de supprimer dans un nuage de points, les poins en dessous d'un seuil en z.

def remove_points_below_threshold(input_ply_file, output_ply_file, z_threshold):
    # Charger le fichier PLY
    point_cloud = o3d.io.read_point_cloud(input_ply_file)

    # Convertir le nuage de points Open3D en un tableau NumPy
    point_cloud_np = np.asarray(point_cloud.points)

    # Sélectionner les indices des points dont la coordonnée Z est supérieure ou égale à z_threshold
    indices = np.where(point_cloud_np[:, 2] >= z_threshold)

    # Créer un nouveau nuage de points à partir des indices sélectionnés
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np[indices])

    # Enregistrer le nuage de points filtré dans un nouveau fichier
    o3d.io.write_point_cloud(output_ply_file, filtered_point_cloud)

# Exemple d'utilisation de la fonction
input_ply_file = "foie_de_boeuf_de_dos.ply"  # Remplacez par votre propre fichier PLY
output_ply_file = "foie_de_boeuf_de_dos_sans_back.ply"  # Fichier de sortie filtré
z_threshold = -0.1  # Valeur de seuil Z à partir de laquelle supprimer les points

remove_points_below_threshold(input_ply_file, output_ply_file, z_threshold)
