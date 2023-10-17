import numpy as np
from plyfile import PlyData

def ply_to_points(file_path):
    # prend un fichier .ply et le converti en nuage de points 
    plydata = PlyData.read(file_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    points = np.vstack((x, y, z)).T

    return points

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
