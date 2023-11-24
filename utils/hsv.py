import numpy as np


def rgb_to_hsv(colors):
    """
    Convertit une liste de couleurs RGB en couleurs HSV.

    Parameters:
    - colors (list): Liste de couleurs RGB sous la forme [[r, g, b], ...] avec r, g, et b entre 0 et 255.

    Returns:
    - numpy.ndarray: Tableau des couleurs HSV.
    """
    colorshsv = np.asarray([[i, i, i] for i in range(len(colors))])
    for i in range(len(colors)):
        r = colors[i][0]/255
        g = colors[i][1]/255
        b = colors[i][2]/255
        maximum = max([r, g, b])
        minimum = min([r, g, b])
        v = maximum
        if (v == 0):
            s = 0
        else:
            s = (maximum-minimum)/maximum

        if (maximum-minimum == 0):
            h = 0
        else:
            if (v == r):
                h = 60*(g-b)/(maximum-minimum)

            if (v == g):
                h = 120 + 60*(b-r)/(maximum-minimum)

            if (v == b):
                h = 240+60*(r-g)/(maximum-minimum)

        if (h < 0):
            h = h+360

        h = h/360
        colorshsv[i][0] = h*255
        colorshsv[i][1] = s*255
        colorshsv[i][2] = v*255
    return colorshsv


def mask(nuagespoints, colorspoints, maskhsv, tableau_indice=[]):
    """
    Filtrer les points d'un nuage en fonction d'un masque HSV.

    Parameters:
    - nuagespoints (numpy.ndarray): Tableau des coordonnées des points.
    - colorspoints (numpy.ndarray): Tableau des couleurs associées aux points.
    - maskhsv (list): Masque HSV sous la forme [[h_min, s_min, v_min], [h_max, s_max, v_max]].
    - tableau_indice (list): Tableau des indices des points.

    Returns:
    - tuple: Tuple contenant les points filtrés, les couleurs filtrées, et les indices filtrés (si fournis).
    """

    points = nuagespoints
    colors = colorspoints
    # Convertir l'image "color" RVB en image "color" HSV
    colorshsv = rgb_to_hsv(colors)
    # Construction du masque
    msk = [False for i in range(0, len(colors))]
    for i in range(0, len(colors)):
        # condition : les trois composantes hsv de l'image doivent être incluse entre les deux valeurs de seuil du masque hsv
        if ((colorshsv[i][0] > maskhsv[0][0]) and (colorshsv[i][0] < maskhsv[1][0])):  # Composante h
            # Composante s
            if ((colorshsv[i][1] > maskhsv[0][1]) and (colorshsv[i][1] < maskhsv[1][1])):
                # Composante v
                if ((colorshsv[i][2] > maskhsv[0][2]) and (colorshsv[i][2] < maskhsv[1][2])):
                    msk[i] = True
    # Filtrage des points, des couleurs et des indices (si nécessaire)
    points = points[msk]
    colors = colors[msk]

    if len(tableau_indice) > 0:
        tableau_indice = tableau_indice[msk]
        return np.array(points), np.array(colors), np.array(tableau_indice)

    return np.array(points), np.array(colors)
