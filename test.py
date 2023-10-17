import tkinter as tk
from tkinter import ttk
from tkinter import Scale, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import csv

def affichage_matplotlib(ax, points):
    ax.cla()
    points = np.array(points) 
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')
    ax.set_title('Nuage de points 3D')

def filtrage_barycentre(points, rayon):
    barycentre = np.mean(points, axis=0)
    rayon_filtrage = rayon
    filtered_points = []
    for point in points:
        distance = np.linalg.norm(point - barycentre)
        if distance <= rayon_filtrage:
            filtered_points.append(point)
    return np.array(filtered_points)

def update_filter(event, ax, canvas, points, rayon_slider):
    rayon = rayon_slider.get()
    filtered_points = filtrage_barycentre(points, rayon)
    affichage_matplotlib(ax, filtered_points)
    canvas.draw()

def export_filtered_data(filtered_data, root):
    if filtered_data:
        filename = "filtered_points.csv"
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(filtered_data)
        print(f"Les points filtrés ont été exportés dans {filename}")
    root.quit()  # Stop the Tkinter event loop

def interface_de_filtrage_de_points(points):
    root = tk.Tk()
    root.title("Filtrage de nuage de points")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    rayon_label = ttk.Label(root, text="Rayon de filtrage")
    rayon_label.pack()
    rayon_slider = Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal", command=lambda event: update_filter(event, ax, canvas, points, rayon_slider))
    rayon_slider.pack()

    affichage_matplotlib(ax, points)

    filtered_data = []

    export_button = Button(root, text="Exporter les points filtrés", command=lambda: export_filtered_data(filtered_data, root))
    export_button.pack()

    root.mainloop()
    
    rayon = rayon_slider.get()
    filtered_points = filtrage_barycentre(points, rayon)
    filtered_data.extend(filtered_points)

    return np.array(filtered_data)

# Exemple de nuage de points
points = np.random.rand(100, 3)

# Appel de la fonction pour créer l'interface et filtrer les points
filtered_points = interface_de_filtrage_de_points(points)