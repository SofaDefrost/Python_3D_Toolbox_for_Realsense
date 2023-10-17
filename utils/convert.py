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

def export_ply(points, filename):
    #Permet d'exporter un nuage de point en .ply

    num_points = points.shape[0]  # Nombre de points
    num_dimensions = points.shape[1]  # Nombre de dimensions par point

    if num_dimensions < 3:
        print("Le format PLY nécessite au moins 3 dimensions (x, y, z).")
        return

    with open(filename, 'w') as ply_file:
        # Écriture de l'en-tête PLY
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(num_points))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # Écriture des coordonnées des points
        for i in range(num_points):
            x, y, z = points[i][:3]  # Assurez-vous que les coordonnées sont 3D (x, y, z)
            ply_file.write(f"{x} {y} {z}\n")

    print(f"Le nuage de points a été exporté en format PLY dans le fichier '{filename}'.")

    