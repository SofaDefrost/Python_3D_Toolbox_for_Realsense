#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:02:23 2023

@author: Tinhinane and Thibaud
"""

import cv2
import numpy as np 

def crop_points_cloud(image_path,points_cloud,couleurs,h,tableau_indice=[]):
    
# Fonction permttant de couper un nuage de point à partir de sa projection 2D (image). h correspond à la longueur de l'image.
# les tableaux doivent être en ligne
# tableau_indice est le tableau qui correspond aux indices du nuage de point intial (avant le crop)

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
            cv2.rectangle(image_copy, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow("Cropping", image_copy)

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur: Impossible de charger l'image. Veuillez vérifier le chemin.")
        return None

    image_copy = image.copy()

    # Créer une fenêtre pour l'image
    cv2.namedWindow("Cropping")
    cv2.setMouseCallback("Cropping", mouse_click)

    # Instructions pour l'utilisateur
    print("Utilisez la souris pour sélectionner le rectangle de recadrage. Appuyez sur la touche 'c' puis 'q' pour terminer le recadrage.")

    # Boucle principale
    while True:
        cv2.imshow("Cropping", image_copy)
        key = cv2.waitKey(1) & 0xFF
        
        # Quitter la boucle si la touche "c" est pressée
        if key == ord("c"):
            break

    # Vérifier si le rectangle de recadrage a une taille valide
    if start_x == end_x or start_y == end_y:
        print("Erreur: Le rectangle de recadrage a une taille invalide.")
        return None

    # Récupérer les coordonnées de cropping
    x_min, y_min = min(start_x, end_x), min(start_y, end_y)
    x_max, y_max = max(start_x, end_x), max(start_y, end_y)

    # Filtrage du nuage de points 
    bottom_left_corner = (y_min-1)*h +x_min
    top_left_corner = (y_max-1)*h +x_min
    bottom_right_corner = (y_min-1)*h +x_max
    top_right_corner = (y_max-1)*h +x_max
    
    i=0    
    points_cloud_crop=[]
    couleurs_crop=[]
    tableau_indice_crop=[]

    while(bottom_left_corner != top_left_corner):
        for j in range(bottom_left_corner,bottom_right_corner):
            points_cloud_crop.append(points_cloud[j])
            couleurs_crop.append(couleurs[j])
            if len(tableau_indice)>0:
                tableau_indice_crop.append(tableau_indice[j])
        bottom_left_corner = (y_min+i-1)*h +x_min
        bottom_right_corner = (y_min+i-1)*h +x_max
        i+=1

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(tableau_indice_crop)>0:
        return np.array(points_cloud_crop),np.array(couleurs_crop),np.array(tableau_indice_crop)
    else:
        return np.array(points_cloud_crop),np.array(couleurs_crop)

