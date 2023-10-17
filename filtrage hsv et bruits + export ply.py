import numpy as np
import acquisition as get
import filtre_hsv_realsense as filtre
from utils import hsv as apply_hsv
from utils import filtrage_bruit as bruit
from utils import convert as cv

# Fichier d'exemple qui montre comment utilser les différentes fonctions. 
# A l'éxecution du programme une acquisition est effectuée, avec un filtrage hsv + filtrage anti bruit
# A la fin est exporté un fichier .ply de l'acquisition

def colors_relasense_sofa(colors):
    # Permet de convertir les couleurs de la caméra realsense  en un format lisible par Sofa (ie on met l'image en ligne)
    l=len(colors)*len(colors[0])
    new_colors=np.asarray([(0,0,0) for i in range(l)])
    indice=0
    for i in range(len(colors)):
        for j in range(len(colors[0])):
            new_colors[indice]=colors[i][j]
            indice+=1
    return np.array(new_colors)

def points_realsense_sofa(vertices):
    # Permet de convertir les points de la caméra realsense en un format lisible par Sofa
    points=[]
    valid_points_list=vertices.tolist()
    for i in range(len(valid_points_list)):
        l=[valid_points_list[i][0],valid_points_list[i][1],valid_points_list[i][2]]
        points.append(l+[0.,0.,0.,0.]) # on crée une liste points qui va stocker les informations, [0.,0.,0.,0.] correspond aux cordonée qx qy qz qw
    return np.array(points)

# On détermine le masque à appliquer
mask=filtre.determinemaskhsv()
# On récupère les points et les couleurs de la caméra
vertices,colors = get.points_and_colors_realsense()
# On converit les couleurs dans un bon format (ie on met l'image en ligne)
colors_sofa = colors_relasense_sofa(colors)
cv.create_ply_file(vertices,colors_sofa,"acquiistion.ply")
# On applique le masque
new_points , new_colors = apply_hsv.mask(vertices,colors_sofa,mask)
new_points_sofa=points_realsense_sofa(new_points) # Attention 
# On extrait la partie que l'on veut
new_points_surface=[sous_liste[:3] for sous_liste in new_points_sofa]
# On filtre le bruit
final_points, final_color = bruit.interface_de_filtrage_de_points(np.array(new_points_surface),new_colors)
# On enregistre on format .ply
cv.create_ply_file(final_points,final_color,"test.ply")