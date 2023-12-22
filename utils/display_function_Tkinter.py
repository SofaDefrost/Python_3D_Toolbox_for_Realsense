import numpy as np
import tkinter as tk
import matplotlib.axes._axes
import matplotlib.pyplot as plt
import sys

from tkinter import ttk
from tkinter import Scale, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Any, Callable, Optional


mod_name = vars(sys.modules[__name__])['__package__']
if mod_name:
    # Code executed as a module
    from . import processing_array as pa
    from . import processing_ply as pl
else:
    # Code executed as a script
    import processing_array as pa
    import processing_ply as pl


def plot_3Darray_Tkinter(ax: matplotlib.axes._axes.Axes, points: np.ndarray) -> None:
    """
    Plot a 3D array.

    Parameters:
    - ax (Axes3D): The 3D axes.
    - points (np.ndarray): The 3D array to be plotted.
    """
    ax.cla()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # Set labels and title
    ax.set_title('3D Array')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')


def update_display_Tkinter(event: Any, ax: Axes3D, canvas: Any, points: np.ndarray, fonction, rayon_slider: Any) -> None:
    """
    Update the display based on the slider value.

    Parameters:
    - event (Any): The event trigger.
    - ax (Axes3D): The 3D axes.
    - canvas (Any): The Tkinter canvas.
    - points (np.ndarray): The 3D array.
    - fonction: The function to be applied.
    - rayon_slider (Any): The slider for the parameter.
    """
    rayon = rayon_slider.get()
    new_points = fonction(
        points, rayon)
    plot_3Darray_Tkinter(ax, new_points)
    canvas.draw()


def get_parameter_function_on_array_Tkinter(points: np.ndarray, fonction: Callable[[np.ndarray, float]], start_slider: int = 0, end_slider: int = 1, resolution: float = 0.01) -> float:
    """
    Get a parameter value interactively using Tkinter GUI.

    Parameters:
    - points (np.ndarray): The 3D array.
    - fonction: The function to be applied.
    - start_slider (float): The starting value for the slider.
    - end_slider (float): The ending value for the slider.
    - resolution (float): The resolution of the slider.

    Returns:
    float: The selected parameter value.
    """
    root = tk.Tk()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    slider_label = ttk.Label(
        root, text=f"{fonction.__code__.co_varnames[:fonction.__code__.co_argcount][1]}")
    slider_label.pack()
    slider = Scale(root, from_=start_slider, to=end_slider, resolution=resolution, orient="horizontal", command=lambda event: update_display_Tkinter(
        event, ax, canvas, points, fonction, slider))
    slider.pack()

    plot_3Darray_Tkinter(ax, points)

    export_button = Button(root, text=f"Export value of the {fonction.__code__.co_varnames[:fonction.__code__.co_argcount][1]}",
                           command=root.quit)
    export_button.pack()

    root.mainloop()

    return slider.get()


def template_function_for_Tkinter_display(points: np.ndarray, parameter_that_you_want: float, optional_other_arguments: Optional[Any] = None) -> np.ndarray:
    """
    Template function for Tkinter display.

    Parameters:
    - points (np.ndarray): Input 3D array of points.
    - parameter_that_you_want (float): The parameter you want to adjust.
    - optional_other_arguments (Any, optional): Additional optional arguments. Default is None.

    Returns:
    np.ndarray: The processed 3D array of points.
    """
    # DO STUFF (below is an example)
    new_points = points  # Example
    # Return 3D np.array
    return np.array(new_points)


if __name__ == '__main__':
    points, colors = pl.get_points_and_colors_of_ply("test.ply")
    print(get_parameter_function_on_array_Tkinter(
        points, pa.reduce_density_of_array))
    print(get_parameter_function_on_array_Tkinter(
        points, pa.filter_array_with_sphere_on_barycentre))
    print(get_parameter_function_on_array_Tkinter(
        points, pa.remove_points_of_array_below_threshold))
