import numpy as np
import pyvista as pv
import cv2
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from typing import List, Optional, Tuple

import processing_float as pfl
import processing_ply as pp
import processing_img as pi
import processing_general as pg

# Arrays

def is_homogenous_array_of_dim(array,dimension=-1)->None:
    if len(array)==0:
        raise ValueError(f"Empty array {array}")
    if not all(isinstance(element, type(array[0])) for element in array):
        raise ValueError(f"Different type in the array {array}")
    if dimension > 0:
        taille=np.shape(array)
        if not(len(taille)==1 and dimension ==1):
            if not taille[1] == dimension:
                raise ValueError(f"Not the correct dimension for the array {array}, dimension expected {dimension}")

def plot_3D_array(vertices: np.ndarray) -> None:
    """
    Plot 3D points and triangles from PLY file vertices.

    Parameters:
    - vertices (np.ndarray): An array containing vertices with X, Y, Z coordinates.
      If the array has 6 columns, it is assumed to include color information (RGB).

    Returns:
    - None
    """
    is_homogenous_array_of_dim(vertices,3)
    # Extract color information if available
    if vertices.shape[1] == 6:
        # Normalize colors to the range [0, 1]
        colors = vertices[:, 3:6] / 255.0
    else:
        colors = 'b'  # Use blue as default color if no color information is present

    # Perform Delaunay triangulation
    triangles = Delaunay(vertices[:, :2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices with colors
    ax.scatter(vertices[:, 0], vertices[:, 1],
               vertices[:, 2], c=colors, marker='o')

    # Plot the triangles
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=triangles.simplices, color='gray', alpha=0.5)

    plt.show()

def build_mesh_from_3Darray(points: np.ndarray) -> np.ndarray:
    """
    Calcule et reconstruit une surface à partir d'un nuage de points.

    Parameters:
    - points (np.array): Nuage de points sous forme d'un tableau NumPy.

    Returns:
    - triangles (np.array): Liste des triangles de la surface reconstruite.
    """
    is_homogenous_array_of_dim(points,3)
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

def add_list_at_each_rows_of_array(array: np.ndarray, list: List) -> np.ndarray:
    is_homogenous_array_of_dim(array)
    new_array = []
    array = array.tolist()
    for i in range(len(array)):
        new_array.append(array[i]+list)
    return np.array(new_array)

def array_to_line(array: np.ndarray) -> np.ndarray:
    is_homogenous_array_of_dim(array)
    return array.reshape((np.shape(array)[0]*np.shape(array)[1],3))

def line_to_3Darray(line:np.ndarray,shape:Tuple[int,int])->np.ndarray:
    is_homogenous_array_of_dim(line)
    return line.reshape((shape[0],shape[1],3))

def give_min_mean_max_of_array(array: np.ndarray) ->  Tuple[float,float,float]:
    """
    Calcule la valeur maximale, minimale et moyenne à partir d'une liste.

    Args:
        list (list): Liste des valeurs à analyser.

    Returns:
        tuple or None: Un tuple contenant la valeur minimale, moyenne et maximale, 
                      ou None si la liste est vide.
    """
    is_homogenous_array_of_dim(array,1)

    array_max = max(array)
    array_min = min(array)
    array_mean = sum(array) / len(array)
    return array_min, array_mean, array_max

def convert_rgb_array_to_hsv_array(colors: np.ndarray) -> np.ndarray:
    """
    Convertit une liste de couleurs RGB en couleurs HSV.

    Parameters:
    - colors (list): Liste de couleurs RGB sous la forme [[r, g, b], ...] avec r, g, et b entre 0 et 255.

    Returns:
    - numpy.ndarray: Tableau des couleurs HSV.
    """
    is_homogenous_array_of_dim(colors,3)
    colorshsv = np.asarray([[i, i, i] for i in range(len(colors))])
    for i in range(len(colors)):
        r = colors[i][0]/255
        g = colors[i][1]/255
        b = colors[i][2]/255
        maximum = max([r, g, b])
        minimum = min([r, g, b])
        v = maximum
        if (v == 0):
            s = 0
        else:
            s = (maximum-minimum)/maximum

        if (maximum-minimum == 0):
            h = 0
        else:
            if (v == r):
                h = 60*(g-b)/(maximum-minimum)

            if (v == g):
                h = 120 + 60*(b-r)/(maximum-minimum)

            if (v == b):
                h = 240+60*(r-g)/(maximum-minimum)

        if (h < 0):
            h = h+360

        h = h/360
        colorshsv[i][0] = h*255
        colorshsv[i][1] = s*255
        colorshsv[i][2] = v*255
    return colorshsv

def get_color_3D_array_depending_on_axis(points: np.ndarray, axis: str) -> np.ndarray:

    is_homogenous_array_of_dim(points,3)
    
    if axis != "x" and axis != "y" and axis != "z":
        raise ValueError(f"Unknown axis: {axis}. Must be x, y or z")

    colors = np.array([[0, 0, 0] for i in range(len(points))])

    if axis == "x":
        axis_int = 0
    if axis == "y":
        axis_int = 1
    if axis == "z":
        axis_int = 2

    axis_coordinates = [coord[axis_int] for coord in points]
    min_coord, mean_coord, max_coord = give_min_mean_max_of_array(
        axis_coordinates)

    for i in range(len(colors)):
        colors[i] = pfl.convert_float_in_range_to_rgb(
            points[i][axis_int], min_coord, mean_coord, max_coord)
    return np.array(colors)

def centering_3Darray_on_mean_points(array):
    """
    Repose the point cloud mesh by centering it around its mean point.

    Parameters
    ----------
    pc_resized : list
        The  point cloud mesh.

    Returns
    -------
    pc_respoed : list
        The list reposed.
"""
    is_homogenous_array_of_dim(array,3)
    tab_x = []
    tab_y = []
    tab_z = []

    for i in array:
        tab_x.append(i[0])
        tab_y.append(i[1])
        tab_z.append(i[2])

    milieu_x = np.mean(tab_x)
    milieu_y = np.mean(tab_y)
    milieu_z = np.mean(tab_z)

    pt_milieu = [milieu_x, milieu_y, milieu_z]

    new_vertices =[point - pt_milieu for point in array]

    return new_vertices

# Arrays and points clouds

def remove_points_of_array_below_threshold(points: np.ndarray,threshold: float,colors:np.ndarray=[],axis:str="z"):

    is_homogenous_array_of_dim(points,3)
    
    if axis != "x" and axis != "y" and axis != "z":
        raise ValueError(f"Unknown axis: {axis}. Must be x, y or z")

    if axis == "x":
        axis_int = 0
    if axis == "y":
        axis_int = 1
    if axis == "z":
        axis_int = 2
    
    # Sélectionner les indices des points dont la coordonnée Z est supérieure ou égale à z_threshold
    index = np.where(points[:,axis_int] >= threshold)

    # Vérifier si le nuage de points filtré n'est pas vide
    if len(index[0]) == 0:
        raise ValueError("Every points have been removed")

    if len(colors)>0:
        new_points=points[index]
        new_colors=colors[index]
        return np.array(new_points),np.array(new_colors)
    else:
        new_points=points[index]
        return np.array(new_points)
       
def reduce_density_of_array(points: np.ndarray,densite: float,colors:np.ndarray=[]):
    
    colors_reduits=[]
    
    # Calculer le nombre de points à conserver
    nombre_points = int(len(points) * densite)

    # Sélectionner aléatoirement les indices des points à conserver
    indices_a_conserver = np.random.choice(
        len(points), nombre_points, replace=False)

    # Extraire les points sélectionnés
    points_reduits = points[indices_a_conserver, :]
    if len(colors) != 0:
        colors_reduits = colors[indices_a_conserver, :]
        return np.array(points_reduits),np.array(colors_reduits)
    
    return np.array(points_reduits)

def filter_array_with_sphere_on_barycentre(points: np.ndarray, rayon: float,colors:np.ndarray=[], tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtre les points en fonction du barycentre et du rayon spécifiés.

    Parameters:
    - points (numpy.ndarray): Tableau des coordonnées des points.
    - colors (numpy.ndarray): Tableau des couleurs associées aux points.
    - rayon (float): Rayon de filtrage autour du barycentre.
    - tableau_indice (list): Tableau des indices des points (optionnel).

    Returns:
    - tuple: Tuple contenant les points filtrés, les couleurs filtrées, et les indices filtrés (si fournis).
    """
    barycentre = np.mean(points, axis=0)
    rayon_filtrage = rayon
    filtered_points = []
    filtered_colors = []
    filtered_indice = []

    for i, point in enumerate(points):
        distance = np.linalg.norm(point - barycentre)
        if distance <= rayon_filtrage:
            filtered_points.append(point)
            if len(colors)>0:
                filtered_colors.append(colors[i])
            if len(tableau_indice) > 0:
                filtered_indice.append(tableau_indice[i])
    if len(tableau_indice) > 0:
                return np.array(filtered_points), np.array(filtered_colors), np.array(filtered_indice)
    if len(colors)>0:
        return np.array(filtered_points), np.array(filtered_colors)
    return np.array(filtered_points)

# Points clouds

def crop_pc_from_zone_selection(points: np.ndarray, colors: np.ndarray,shape:Tuple[int,int]=[], tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    # Variables globales pour stocker les coordonnées des clics souris
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    cropping = False

    def mouse_click(event, x, y, flags, param):
        # Référence aux variables globales
        nonlocal start_x, start_y, end_x, end_y, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            # Début du cropping
            start_x, start_y = x, y
            end_x, end_y = x, y
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            # Fin du cropping
            end_x, end_y = x, y
            cropping = False
            # Dessine le rectangle de cropping
            cv2.rectangle(colors_image, (start_x, start_y),
                          (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow("Cropping", colors_image)

    if shape!=[]:
        if np.shape(shape)!=(2,):
            raise ValueError(f"Incorrect shape {shape} for the display")
        colors_image=line_to_3Darray(colors,(height,length)).astype(np.uint8)
        length,height=shape
    else:
        if len(np.shape(colors))!=3:
            raise ValueError(f"Incorrect shape for the display got {np.shape(colors)} and expected (x,y,z)")
        if np.shape(colors)[2]!=3:
            raise ValueError(f"Incorrect dimension for the array, expected 3 and given {np.shape(colors)[2]}")
        colors_image=colors.astype(np.uint8)
        colors=array_to_line(colors)
        height,length=np.shape(colors_image)[0],np.shape(colors_image)[1]
    
    # Créer une fenêtre pour l'image
    cv2.namedWindow("Cropping")
    cv2.setMouseCallback("Cropping", mouse_click)

    # Instructions pour l'utilisateur
    print("Utilisez la souris pour sélectionner le rectangle de recadrage. Appuyez sur la touche 'c' puis 'q' pour terminer le recadrage.")

    # Boucle principale
    while True:
        cv2.imshow("Cropping", colors_image[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF

        # Quitter la boucle si la touche "c" est pressée
        if key == ord("c"):
            break

    # Vérifier si le rectangle de recadrage a une taille valide
    if start_x == end_x or start_y == end_y:
        raise ValueError("Incorrect values for cropping selected")

    # Récupérer les coordonnées de cropping
    x_min, y_min = min(start_x, end_x), min(start_y, end_y)
    x_max, y_max = max(start_x, end_x), max(start_y, end_y)

    # Filtrage du nuage de points
    bottom_left_corner = (y_min-1)*length + x_min
    top_left_corner = (y_max-1)*length + x_min
    bottom_right_corner = (y_min-1)*length + x_max

    i = 0
    points_cloud_crop = []
    couleurs_crop = []
    tableau_indice_crop = []

    while (bottom_left_corner != top_left_corner):
        for j in range(bottom_left_corner, bottom_right_corner):
            points_cloud_crop.append(points[j])
            couleurs_crop.append(colors[j])
            if len(tableau_indice) > 0:
                tableau_indice_crop.append(tableau_indice[j])
        bottom_left_corner = (y_min+i-1)*length + x_min
        bottom_right_corner = (y_min+i-1)*length + x_max
        i += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(tableau_indice_crop) > 0:
        return np.array(points_cloud_crop), np.array(couleurs_crop), np.array(tableau_indice_crop)
    else:
        return np.array(points_cloud_crop), np.array(couleurs_crop)

def apply_hsv_mask_to_pc(points: np.ndarray, colors: np.ndarray, maskhsv: List[List[int]], tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    # Convertir l'image "color" RVB en image "color" HSV
    colorshsv = convert_rgb_array_to_hsv_array(colors)
    # Construction du masque
    msk = [False for i in range(0, len(colors))]
    for i in range(0, len(colors)):
        # condition : les trois composantes hsv de l'image doivent être incluse entre les deux valeurs de seuil du masque hsv
        if ((colorshsv[i][0] > maskhsv[0][0]) and (colorshsv[i][0] < maskhsv[1][0])):  # Composante h
            # Composante s
            if ((colorshsv[i][1] > maskhsv[0][1]) and (colorshsv[i][1] < maskhsv[1][1])):
                # Composante v
                if ((colorshsv[i][2] > maskhsv[0][2]) and (colorshsv[i][2] < maskhsv[1][2])):
                    msk[i] = True
    
    # Filtrage des points, des couleurs et des indices (si nécessaire)
    points = points[msk]
    colors = colors[msk]

    if len(tableau_indice) > 0:
        tableau_indice = tableau_indice[msk]
        return np.array(points), np.array(colors), np.array(tableau_indice)

    return np.array(points), np.array(colors)

def center_pc_on_image(points: np.ndarray,colors:np.ndarray,image_target: str, shape:Tuple[int,int]=[]) -> Tuple[np.ndarray, np.ndarray]:

    largeur,hauteur=pi.get_size_of_image(image_target)

    centre_image_ref = np.array(
        [int(largeur/2), int(hauteur/2), 1])  # En pixel

    image_source="center_pc_on_image.png"
    
    pi.save_image_from_array(colors,image_source,shape)
    
    longueur,_=pi.get_size_of_image(image_source)
    
    # On calcule les coordonnées (toujours en pixel) de ce centre dans l'image source
    # Pour cela on calcule la matrice d'homographie
    homography_matrix = pi.get_homography_between_imgs(image_target, image_source)
    pg.delete_file(image_source)
    
    projected_center = np.dot(homography_matrix, centre_image_ref)  # en pixel

    # On repasse en coordonnées en profondeur

    origine_nuage = points[(int(projected_center[1])-1)
                             * longueur + int(projected_center[0])]

    # On repose ensuite le nuage de point pour que ce dernier soit centré autour du point origine_nuage
    # (qui correpont au centre de l'objet de l'image de reference)

    nuage_point_centred = [(x[0] - origine_nuage[0], x[1] -
                            origine_nuage[1], x[2] - origine_nuage[2]) for x in points]

    return nuage_point_centred, colors

if __name__ == '__main__':
    l=np.array([1,2,3])
    points, colors = pp.get_points_and_colors_of_ply("test.ply")
    crop_pc_from_zone_selection(points,colors,640,480)
    # new_colors=line_to_3Darray(colors,(480,640))
    # pi.save_PIL_image_from_array(new_colors,"oui.png")
    # plot_3D_array(points)
    # print(add_list_at_each_rows_of_array(points, [0., 0., 0., 1.])[0])
    # print(np.shape(colors))
    # print(array_to_line(colors))
    # test=[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]
    # liste_triangles=build_mesh_from_3Darray(test)
