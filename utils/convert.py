import numpy as np
import open3d as o3d

def ply_to_points_and_colors(file_path):
    # prend un fichier .ply et le converti en nuage de points et en couleur format RGB entre 0 et 255
    ply_data = o3d.io.read_point_cloud(file_path)
    points = np.array(ply_data.points)
    colors = np.array(ply_data.colors)* 255

    return points, colors

def create_ply_file(points, colors, output_filename):
    if len(points) != len(colors):
        raise ValueError("Le nombre de points doit correspondre au nombre de couleurs.")
    
    with open(output_filename, 'w') as ply_file:
        # Écriture de l'en-tête PLY
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(len(points)))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        # Écriture des données de points et couleurs
        for point, color in zip(points, colors):
            x, y, z = point
            r, g, b = color
            ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")
            
        print(f"Le fichier '{output_filename}' a été créé.")

def create_ply_file_without_colors(points, output_filename):

    # Ouvre le fichier de sortie en mode écriture.
    with open(output_filename, 'w') as ply_file:
        # Écrit l'en-tête PLY.
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(len(points)))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # Écrit les coordonnées des points dans le fichier.
        for point in points:
            ply_file.write("{} {} {}\n".format(point[0], point[1], point[2]))


