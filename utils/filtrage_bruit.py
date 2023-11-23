import tkinter as tk
from tkinter import ttk
from tkinter import Scale, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import csv

def affichage_matplotlib(ax, points):
    ax.cla()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')
    ax.set_title('Nuage de points 3D')

def filtrage_barycentre(points,colors, rayon,tableau_indice):
    barycentre = np.mean(points, axis=0)
    rayon_filtrage = rayon
    filtered_points = []
    filtered_colors = []
    filtered_indice=[]
    for i in range(len(points)):
        point=points[i]
        distance = np.linalg.norm(point - barycentre)
        if distance <= rayon_filtrage:
            filtered_points.append(point)
            filtered_colors.append(colors[i])
            if len(tableau_indice)>0:
                filtered_indice.append(tableau_indice[i])

    return np.array(filtered_points), np.array(filtered_colors),np.array(filtered_indice)

def update_filter(event, ax, canvas, points, colors, rayon_slider,tableau_indice):
    rayon = rayon_slider.get()
    filtered_points , filtered_colors,filtered_tableau_indice = filtrage_barycentre(points,colors, rayon,tableau_indice)
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

def interface_de_filtrage_de_points(points,colors,tableau_indice=[]):
    root = tk.Tk()
    root.title("Filtrage de nuage de points")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    rayon_label = ttk.Label(root, text="Rayon de filtrage")
    rayon_label.pack()
    rayon_slider = Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal", command=lambda event: update_filter(event, ax, canvas, points, colors, rayon_slider,tableau_indice))
    rayon_slider.pack()

    affichage_matplotlib(ax, points)

    filtered_data_points = []
    filtered_data_colors = []
    filtered_data_indices = []

    export_button = Button(root, text="Exporter les points filtrés", command=lambda: export_filtered_data(filtered_data_points, root))
    export_button.pack()

    root.mainloop()
    
    rayon = rayon_slider.get()
    filtered_points, filtered_colors,filtered_indices = filtrage_barycentre(points,colors, rayon,tableau_indice)
    filtered_data_points.extend(filtered_points)
    filtered_data_colors.extend(filtered_colors)
    
    if len(filtered_indices)>0:
        filtered_data_indices.extend(filtered_indices)
        return np.array(filtered_data_points), np.array(filtered_data_colors),np.array(filtered_data_indices)
    else:
        return np.array(filtered_data_points), np.array(filtered_data_colors)

# # Exemple de nuage de points avec ses couleurs
# points = np.random.rand(100, 3)
# colors = np.random.randint(0, 256, size=(100, 3))

# # Appel de la fonction pour créer l'interface et filtrer les points
# filtered_points, filtered_colors = interface_de_filtrage_de_points(points,colors)
# print(filtered_colors)
# print(filtered_points)
