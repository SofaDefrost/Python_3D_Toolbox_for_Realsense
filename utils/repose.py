#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 20:29:42 2023

@author: tinhinane
"""

import numpy as np
import trimesh 

def repose(pc_resized, pc_reposed):
    """
    Repose the point cloud mesh by centering it around its mean point.

    Parameters
    ----------
    pc_resized : str
        The filename of the resized point cloud mesh in PLY format.
    pc_reposed : str
        The filename to save the reposed point cloud mesh.

    Returns
    -------
    pt_milieu : list
        The coordinates of the mean point before reposing.
"""
    
    mesh = trimesh.load(pc_resized)

    tab_x = []
    tab_y = []
    tab_z = []

    for i in mesh.vertices:
        tab_x.append(i[0])
        tab_y.append(i[1])
        tab_z.append(i[2])

    milieu_x = np.mean(tab_x)
    milieu_y = np.mean(tab_y)
    milieu_z = np.mean(tab_z)

    pt_milieu = [milieu_x, milieu_y, milieu_z]

    new_vertices = mesh.vertices - pt_milieu
    scaled_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces if hasattr(mesh, 'faces') else None)

    scaled_mesh.export(pc_reposed)
    return pt_milieu

def repose_points(points):
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
    tab_x = []
    tab_y = []
    tab_z = []

    for i in points:
        tab_x.append(i[0])
        tab_y.append(i[1])
        tab_z.append(i[2])

    milieu_x = np.mean(tab_x)
    milieu_y = np.mean(tab_y)
    milieu_z = np.mean(tab_z)

    pt_milieu = [milieu_x, milieu_y, milieu_z]

    new_vertices =[point - pt_milieu for point in points]

    return new_vertices

def repose_obj(pc_resized, pc_reposed):
    """
    Repose the point cloud mesh by centering it around its mean point.

    Parameters
    ----------
    pc_resized : str
        The filename of the resized point cloud mesh in OBJ format.
    pc_reposed : str
        The filename to save the reposed point cloud mesh in OBJ format.

    Returns
    -------
    pt_milieu : list
        The coordinates of the mean point before reposing.
    """
    mesh = trimesh.load_mesh(pc_resized)  # Charge le fichier OBJ

    # Calcul du point moyen
    mean_point = np.mean(mesh.vertices, axis=0)
    pt_milieu = mean_point.tolist()

    # Centrage autour du point moyen
    new_vertices = mesh.vertices - mean_point
    reposed_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)

    # Exporte le fichier repositionné au format OBJ
    reposed_mesh.export(pc_reposed)

    return pt_milieu

# import convert as cv
# # Pour repose les fichiers issus du spectromètres

# p,c=cv.ply_to_points_and_colors("foie_V_couleurs_h.ply")
# pr=repose_points(p)
# c=np.array([[int(pixels) for pixels in couleur] for couleur in c])
# cv.create_ply_file(pr,c,"foie_V_reposed.ply")