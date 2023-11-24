import numpy as np


def points_realsense_sofa(vertices):
    """
    Convertit les points de la caméra RealSense en un format lisible par Sofa.

    Parameters:
    - vertices (numpy.ndarray): Tableau des coordonnées XYZ des points de la caméra RealSense.

    Returns:
    - numpy.ndarray: Tableau des points formatés avec des colonnes supplémentaires pour qx, qy, qz, qw.
    """
    points = []
    valid_points_list = vertices.tolist()
    for i in range(len(valid_points_list)):
        l = [valid_points_list[i][0], valid_points_list[i]
             [1], valid_points_list[i][2]]
        # on crée une liste points qui va stocker les informations, [0.,0.,0.,0.] correspond aux cordonée qx qy qz qw
        points.append(l+[0., 0., 0., 0.])
    return np.array(points)


def colors_relasense_sofa(colors):
    """
    Convertit les couleurs de la caméra RealSense en un format lisible par Sofa.

    Parameters:
    - colors (numpy.ndarray): Tableau des couleurs de la caméra RealSense.

    Returns:
    - numpy.ndarray: Tableau des couleurs formaté pour Sofa.
    """
    l = len(colors)*len(colors[0])
    new_colors = np.asarray([(0, 0, 0) for i in range(l)])
    indice = 0
    for i in range(len(colors)):
        for j in range(len(colors[0])):
            new_colors[indice] = colors[i][j]
            indice += 1
    return np.array(new_colors)
