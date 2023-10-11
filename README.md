## RealSense

Ce dépôt fourni tout un ensemble de fonctions utiles pour les caméras Intel Realsense (acquiistion, filtre)

# Organisation du code
*(voir la partie ci-après pour des explications sur le fonctionnement des fonctions)*

Le fichier *acquisition.py* contient les fonctions *run_acquisition* et *points_and_colors_realsense*.

Le fichier *filtre_hsv_realsense.py* contient la fonction *determinemaskhsv*.

Le fichier *main.py* effectue une démonstration de ces fonctions.

# Mode d'emploi et explications

Pour tester le code, il faut exécuter le code python *main.py* :

```
consolepython3 main.py
```

Trois démonstrations vont alors s'effectuer :

La première (qui illustre la fonction *run_acquisition*) permet de faire une acquisition de la caméra Realsense D405 et de l'exporter au format *.ply*. A l'exécution de la fonction, un retour de ce que voit la caméra apparaît en temps réel. L'utilisateur devra ensuite cliquer sur la touche 's' pour enregistrer le nuage de points (au format *.ply*) ainsi que l'image optique (comprenez une image 2D de ce que voit la caméra à l'instant de la capture), et sur la touche 'q' pour arrêter l'acquisition. Notez que l'utilisateur devra fournir en argument de cette fonction un nom de fichier pour le fichier *.ply* ainsi que pour l'image optique (ces noms sont des variables modifiables dans le fichier *main.py*). Une fois l'acquisition terminée, les fichiers *.ply* et l'image optique seront enregistrés dans le même dossier.

La seconde (qui illustre le fonction *points_and_colors_realsense*) permet de faire également une acquisition de la caméra Realsense D405 mais avec quelques différences par rapport à la version précédente. Premièrement, aucune interface visuelle n'apparaît à l'écran de l'utilisateur : l'acquisition se déroule sans. Deuxièmement, l'utilisateur n'a plus besoin d'appuyer sur un bouton pour effectuer la capture, elle se fait de manière automatique et unique. Enfin, aucun fichier (*.ply* ou *.png*) n'est exporté à la fin d'exécution de la fonction : les résultats sont renvoyés sous la forme de deux tableaux, l'un représentant les points dans l'espace (sous la forme d'une liste de taille [307200 ,3]) et l'autre représentant les couleurs associées à ces points (sous la forme d'une liste de taille [480,640, 3])  

La dernière (qui illustre la fonction *determinemaskhsv*) permet de déterminer un masque hsv grâce à la caméra Realsense D405. A l'exécution de cette fonction une interface s'ouvre avec plusieurs fenêtres et notamment une qui permet à l'utilisateur de choisir le masque à l'aide de curseurs. Au fur et à mesure de sa configuration, l'utilisateur voit le retour de l'application de son masque sur une autre fenêtre. Une fois que ce dernier est satisfait de son choix, il peut l'exporter en appuyant sur la touche 'q'. Notez bien qu'aucun filtrage n'est appliqué, la fonction permet juste de déterminer et d'exporter un masque hsv.

## Prérecquis :
Librairies python nécessaires : pyrealsense2 et cv2
```console
pip3 install pyrealsense2
pip3 install opencv-python
```
