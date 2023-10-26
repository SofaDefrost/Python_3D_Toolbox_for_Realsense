import numpy as np

def points_realsense_sofa(vertices):
    # Permet de convertir les points de la caméra realsense  en un format lisible par Sofa
    points=[]
    valid_points_list=vertices.tolist()
    for i in range(len(valid_points_list)):
        l=[valid_points_list[i][0],valid_points_list[i][1],valid_points_list[i][2]]
        points.append(l+[0.,0.,0.,0.]) # on crée une liste points qui va stocker les informations, [0.,0.,0.,0.] correspond aux cordonée qx qy qz qw
    return np.array(points)

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