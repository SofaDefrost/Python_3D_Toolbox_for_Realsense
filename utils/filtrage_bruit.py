import numpy as np
import tkinter as tk
import matplotlib.axes._axes
import matplotlib.pyplot as plt
import csv

from tkinter import ttk
from tkinter import Scale, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import List, Optional, Tuple, Any


def affichage_matplotlib(ax: matplotlib.axes._axes.Axes, points: np.ndarray) -> None:
    """
    Affiche un nuage de points en 3D à l'aide de Matplotlib.

    Parameters:
    - ax (matplotlib.axes._axes.Axes): Instance d'Axes3D de Matplotlib.
    - points (numpy.ndarray): Nuage de points en 3D.
    """
    ax.cla()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # Définir le titre et les étiquettes des axes
    ax.set_title('Nuage de points 3D')
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')


def filtrage_barycentre(points: np.ndarray, colors: np.ndarray, rayon: float, tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            filtered_colors.append(colors[i])
            if len(tableau_indice) > 0:
                filtered_indice.append(tableau_indice[i])

    return np.array(filtered_points), np.array(filtered_colors), np.array(filtered_indice)


def update_filter(event: Any, ax: Axes3D, canvas: Any, points: np.ndarray, colors: np.ndarray, rayon_slider: Any, tableau_indice: list) -> None:
    """
    Met à jour l'affichage en fonction de la valeur du slider de rayon.

    Parameters:
    - event: L'événement déclenchant la mise à jour.
    - ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Instance d'Axes3D de Matplotlib.
    - canvas: L'objet Canvas Matplotlib.
    - points (numpy.ndarray): Tableau des coordonnées des points.
    - colors (numpy.ndarray): Tableau des couleurs associées aux points.
    - rayon_slider: Le widget de slider contrôlant le rayon.
    - tableau_indice (list): Tableau des indices des points.

    Returns:
    - None
    """
    rayon = rayon_slider.get()
    filtered_points, _, _ = filtrage_barycentre(
        points, colors, rayon, tableau_indice)
    affichage_matplotlib(ax, filtered_points)
    canvas.draw()


def export_filtered_data(filtered_data: List[List[str]], root: tk.Tk) -> None:
    """
    Exporte les données filtrées vers un fichier CSV.

    Parameters:
    - filtered_data (list): Liste des données filtrées à exporter.
    - root: L'objet Tkinter racine.

    Returns:
    - None
    """
    if filtered_data:
        filename = "filtered_points.csv"
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(filtered_data)
        print(f"Les points filtrés ont été exportés dans {filename}")
    root.quit()  # Stop the Tkinter event loop


def interface_de_filtrage_de_points(points: np.ndarray, colors: np.ndarray, tableau_indice: Optional[List[int]] = []) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Interface graphique pour filtrer un nuage de points en fonction d'un rayon.

    Parameters:
    - points (numpy.ndarray): Tableau des coordonnées des points.
    - colors (numpy.ndarray): Tableau des couleurs associées aux points.
    - tableau_indice (list): Tableau des indices des points.

    Returns:
    - tuple: Tuple contenant les points filtrés, les couleurs filtrées, et les indices filtrés (si fournis).
    """
    root = tk.Tk()
    root.title("Filtrage de nuage de points")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    rayon_label = ttk.Label(root, text="Rayon de filtrage")
    rayon_label.pack()
    rayon_slider = Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal", command=lambda event: update_filter(
        event, ax, canvas, points, colors, rayon_slider, tableau_indice))
    rayon_slider.pack()

    affichage_matplotlib(ax, points)

    filtered_data_points = []
    filtered_data_colors = []
    filtered_data_indices = []

    export_button = Button(root, text="Exporter les points filtrés",
                           command=lambda: export_filtered_data(filtered_data_points, root))
    export_button.pack()

    root.mainloop()

    rayon = rayon_slider.get()
    filtered_points, filtered_colors, filtered_indices = filtrage_barycentre(
        points, colors, rayon, tableau_indice)
    filtered_data_points.extend(filtered_points)
    filtered_data_colors.extend(filtered_colors)

    if len(filtered_indices) > 0:
        filtered_data_indices.extend(filtered_indices)
        return np.array(filtered_data_points), np.array(filtered_data_colors), np.array(filtered_data_indices)
    else:
        return np.array(filtered_data_points), np.array(filtered_data_colors)

# # Exemple de nuage de points avec ses couleurs
# POINTS = np.random.rand(100, 3)
# COLORS = np.random.randint(0, 256, size=(100, 3))

# # Appel de la fonction pour créer l'interface et filtrer les points
# FILTERED_POINTS, FILTERED_COLORS = interface_de_filtrage_de_points(POINTS,COLORS)
# print(FILTERED_COLORS)
# print(FILTERED_POINTS)
