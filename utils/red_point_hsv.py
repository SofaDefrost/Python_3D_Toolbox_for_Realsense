import cv2

# Fonction qui retourne les coordonnées en pixel d'un point rouge présent dans une image à partir d'un masque hsv

def detect_point_rouge_hsv(path_image,lower_red,upper_red,Affichage=False):

    # Charger l'image
    image = cv2.imread(path_image)  

    # Convertir l'image en espace de couleur HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Créer un masque en utilisant la plage de couleurs définie
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Trouver les contours dans le masque
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_sum=0
    y_sum=0
    sum=0
    # Parcourir tous les contours détectés
    for contour in contours:
        # Récupérer les coordonnées des pixels dans le contour
        for point in contour:
            x, y = point[0]
            x_sum+=x
            y_sum+=y
            sum+=1
    pixel_x=int(x_sum/sum)
    pixel_y=int(y_sum/sum)

    if Affichage:
        # Dessiner les contours sur l'image originale
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Afficher l'image avec les contours
        cv2.imshow('Image avec contours du point laser', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pixel_x,pixel_y