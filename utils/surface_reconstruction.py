import numpy as np
import pyvista as pv


def surface_reconstruction(points: np.ndarray) -> np.ndarray:
    """
    Calcule et reconstruit une surface à partir d'un nuage de points.

    Parameters:
    - points (np.array): Nuage de points sous forme d'un tableau NumPy.

    Returns:
    - triangles (np.array): Liste des triangles de la surface reconstruite.
    """

    cloud = pv.PolyData(points)
    cloud.plot(point_size=15)

    surf = cloud.delaunay_2d()

    # Access the list of triangles
    triangles = surf.faces.tolist()

    liste_indice_triangles = []
    indice = 1
    while indice <= len(triangles)-3:
        liste_indice_triangles.append(
            [triangles[indice], triangles[indice+1], triangles[indice+2]])
        indice = indice + 4

    # Display the list of triangles

    # C'est la liste des triangles qui nous interesse
    liste_triangles = [[points[i] for i in indices_list]
                       for indices_list in liste_indice_triangles]
    surf.plot(show_edges=True)

    return np.array(liste_triangles)

# # Nuage de points d'exemple

# l=[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,0,1],[1,1,1],[0,1,1],[0,0,1],[0.5,0.5,2]] # Maison
# d = [[0,0,0],[4,0,0],[0,4,0],[4,4,0],[0,0,4],[4,0,4],[0,4,4],[4,4,4],[1,1,0],[1,3,0],[3,1,0],[3,3,0],[1,1,4],[1,3,4],[3,1,4],[3,3,4]] # Donut
# test=[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]

# liste_triangles=surface_reconstruction(test)
# print(test)
