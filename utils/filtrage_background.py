import numpy as np
import open3d as o3d


def remove_points_below_threshold(input_ply_file: str, output_ply_file: str, z_threshold: float) -> None:
    """
    Supprime les points d'un nuage de points dont la coordonnée Z est inférieure à un seuil donné.

    Parameters:
    - input_ply_file (str): Chemin du fichier PLY d'entrée.
    - output_ply_file (str): Chemin du fichier PLY de sortie filtré.
    - z_threshold (float): Valeur de seuil Z à partir de laquelle supprimer les points.
    """
    # Vérifier l'existence des fichiers d'entrée
    try:
        with open(input_ply_file):
            pass
    except FileNotFoundError:
        print(f"Erreur: Le fichier d'entrée '{input_ply_file}' n'existe pas.")
        return

    # Charger le fichier PLY
    point_cloud = o3d.io.read_point_cloud(input_ply_file)

    # Convertir le nuage de points Open3D en un tableau NumPy
    points_np = np.asarray(point_cloud.points)
    colors_np = np.asarray(point_cloud.colors)

    # Sélectionner les indices des points dont la coordonnée Z est supérieure ou égale à z_threshold
    indices = np.where(points_np[:, 2] >= z_threshold)

    # Vérifier si le nuage de points filtré n'est pas vide
    if len(indices[0]) == 0:
        print("Aucun point ne satisfait la condition de seuil Z. Le nuage de points filtré est vide.")
        return

    # Créer un nouveau nuage de points à partir des indices sélectionnés
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(
        points_np[indices])
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(
        colors_np[indices])

    # Enregistrer le nuage de points filtré dans un nouveau fichier
    o3d.io.write_point_cloud(output_ply_file, filtered_point_cloud)
    print(
        f"Le nuage de points filtré a été enregistré sous '{output_ply_file}'.")


# Exemple d'utilisation de la fonction
# Remplacez par votre propre fichier PLY
INPUT_PLY_FILE = "foie_de_boeuf_de_dos.ply"
OUTPUT_PLY_FILE = "foie_de_boeuf_de_dos_sans_back.ply"  # Fichier de sortie filtré
Z_THRESHOLD = -0.1  # Valeur de seuil Z à partir de laquelle supprimer les points

remove_points_below_threshold(INPUT_PLY_FILE, OUTPUT_PLY_FILE, Z_THRESHOLD)
