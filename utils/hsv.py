import numpy as np

def rgbtohsv(colors):
    # Permet de convertir une liste de couleur rgb en couleur hsv
    # Le fichier d'entrée doit être sous la forme [[r,g,b],...] avec r, g  et b entre 0 et 255
    colorshsv=np.asarray([[i,i,i] for i in range(len(colors))])
    for i in range(len(colors)):
        r=colors[i][0]/255
        g=colors[i][1]/255
        b=colors[i][2]/255
        maximum=max([r,g,b])
        minimum=min([r,g,b])
        v= maximum
        if (v==0):
            s=0
        else:
            s=(maximum-minimum)/maximum
           
        if(maximum-minimum==0):
            h=0
        else:
            if (v==r):
                h=60*(g-b)/(maximum-minimum)
            
            if (v==g):
                h=120+ 60*(b-r)/(maximum-minimum)
                
            if (v==b):
                h=240+60*(r-g)/(maximum-minimum)

        if(h<0):
            h=h+360

        h=h/360
        colorshsv[i][0]=h*255
        colorshsv[i][1]=s*255
        colorshsv[i][2]=v*255
    return colorshsv  

def mask(nuagespoints, colorspoints, maskhsv):
    # Permet de filtrer les points du nuages en fonction du masque hsv, les points et les couleurs doivent être sous fome de ligne
    points = nuagespoints
    colors = colorspoints
    # Convertir l'image "color" RVB en image "color" HSV 
    colorshsv=rgbtohsv(colors)
    # Construction du masque
    mask=[False for i in range(0,len(colors))]
    for i in range(0,len(colors)):
        # condition : les trois composantes hsv de l'image doivent être incluse entre les deux valeurs de seuil du masque hsv
        if ((colorshsv[i][0] > maskhsv[0][0]) and (colorshsv[i][0] < maskhsv[1][0])): # Composante h
            if((colorshsv[i][1] > maskhsv[0][1]) and (colorshsv[i][1] < maskhsv[1][1])): # Composante s
                if((colorshsv[i][2] > maskhsv[0][2]) and (colorshsv[i][2] < maskhsv[1][2])): # Composante v
                    mask[i]=True
    # Filtrage des points et des couleurs
    points = points[mask]
    colors = colors[mask]
    return np.array(points), np.array(colors)
