# RealSense

Ce dépôt fourni tout un ensemble de fonctions utiles pour les caméras Intel Realsense (acquisition, filtrage...) ainsi qu'un répertoire *Utils* qui regroupe un ensemble de script python utiles en computer vison mais dont l'usage n'est pas reservé aux travaux sur les caméra RealSense. 

## Organisation du dépôt

### Dossier *realsense*

Le fichier *acquisition.py* contient les fonctions qui permettent de faire des acquiistions.

Le fichier *filtre_hsv_realsense.py*  contient les fonctions permettant de déterminer un masque hsv avec une interface adaptée pour la realsense.

Le fichier *get_pc_centered_on.py*  contient les fonctions permettant faire une acquisition et recentrer le nuage de points acquis par rapport au centre de l'objet de référence.

Le fichier *exemple.py* est un fichier d'exemple qui permet à partir d'une caméra RealSense de faire une acquisition, d'effectuer un filtrage hsv et anti bruit et d'exporter le tout au format *.ply*.

### Dossier *realsense/utils*

Le fichier *affichage_ply.py* contient les fonctions permettant de d'afficher un nuage de point dans une interface matplotlib.

Le fichier *color_hauteur.py* contient les fonctions permettant de colorer un fichier *.ply* en fonction de la hauteur de ses points.

Le fichier *convert.py* contient plusieurs fonctions permettant d'effectuer des convertions (exemples: *.ply* -> np.darray, np.darray -> *.ply*...)

Le fichier *crop_points_cloud.py* contient les fonctions permettant de couper un nuage de points à l'aide de la projection 2D d'une image et de points définis à la souris par l'utilisateur.

Le fichier *filtrage_background.py* contient les fonctions permettant de supprimer les points d'un nuage de points dont la coordonnée Z est inférieure à un seuil donné.

Le fichier *filtrage_bruit.py* contient des fonctions permettant de créer une interface de filtrage d'un nuage de point par rapport à son centre de masse.

Le fichier *homography.py* contient des fonctions permettant de trouver la matrice d'homographie entre deux images en utilisant la correspondance de caractéristiques.

Le fichier *hsv.py* contient des fonctions permettant d'appliquer un masque à un nuage de points.

Le fichier *interface_hsv_image.py* contient des fonctions permettant de déterminer un masque hsv à partir d'une image.

Le fichier *realsense_pc.py* contient un ensemble de fonctions permettant de convertir les données d'une realsense dans un format lisible .(notamment) par Sofa.

Le fichier *red_point.py* contient les fonctions qui permettent de détecter un point rouge dans une image.

Le fichier *reduction_densite_pc.py*

Le fichier *repose.py* contient les fonctions qui permettent de recentrer un nuage de points par rapport à son centre de masse.

Le fichier *surface_reconstruction.py* contient les fonctions qui permettent de calculer et de reconstruitre une surface à partir d'un nuage de points.

## Mode d'emploi

Pour tester le code, il faut exécuter le code python *main.py* :
exemple
```console
python3 exemple.py
```

## Prérequis :
Librairies python nécessaires pour l'ensemble du dépôt : numpy, matplotlib, cv2, typing, time, trimesh, pyvista, pyrealsense2, scipy, open3d, tkinter, pil
```console
pip3 install numpy
pip3 install matplotlib
pip3 install opencv-python
pip3 install scipy
pip3 install typing
pip3 install time
pip3 install trimesh
pip3 install pyrealsense2
pip3 install pyvista
pip3 install open3d
pip3 install tkinter
pip3 install pil
```

Pour avoir d'autres informations concernant la realsense avec Python : https://dev.intelrealsense.com/docs/python2

Auteurs : Thibaud, Tinhinane
