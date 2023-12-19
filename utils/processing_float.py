
from typing import Tuple

def convert_float_in_range_to_rgb(float: float, plage_basse: float, plage_moyenne: float, plage_elevee: float) -> Tuple[int, int, int]:
    """
    Convertit un nombre en valeurs RGB en fonction des plages spécifiées.

    Args:
        float (float): Le nombre à convertir.
        plage_basse (float): La limite inférieure de la plage basse.
        plage_moyenne (float): La limite supérieure de la plage basse et la limite 
                              inférieure de la plage moyenne.
        plage_elevee (float): La limite supérieure de la plage moyenne et la limite 
                              inférieure de la plage élevée.

    Returns:
        tuple: Un tuple représentant les valeurs RGB de la couleur correspondant à la hauteur.
    """
    if not (plage_basse <= plage_moyenne and plage_moyenne <= plage_elevee):
        raise ValueError("Incorrect values given for the range")
    
    couleur_basse = (0, 0, 255)  # Bleu pour les chiffres bas dans la plage
    couleur_moyenne = (0, 255, 0)  # Vert pour les chiffres moyen dans la plage
    couleur_elevee = (255, 0, 0)  # Rouge pour les chiffres haut dans la plage

    # Convertir le float en valeurs RGB
    if float <= plage_basse:
        return couleur_basse
    elif plage_basse < float <= plage_moyenne:
        ratio = (float - plage_basse) / (plage_moyenne - plage_basse)
        couleur = (
            int((1 - ratio) * couleur_basse[0] + ratio * couleur_moyenne[0]),
            int((1 - ratio) * couleur_basse[1] + ratio * couleur_moyenne[1]),
            int((1 - ratio) * couleur_basse[2] + ratio * couleur_moyenne[2])
        )
        return couleur
    elif plage_moyenne < float <= plage_elevee:
        ratio = (float - plage_moyenne) / (plage_elevee - plage_moyenne)
        couleur = (
            int((1 - ratio) * couleur_moyenne[0] + ratio * couleur_elevee[0]),
            int((1 - ratio) * couleur_moyenne[1] + ratio * couleur_elevee[1]),
            int((1 - ratio) * couleur_moyenne[2] + ratio * couleur_elevee[2])
        )
        return couleur
    else:
        return couleur_elevee

if __name__ == '__main__':
    print(convert_float_in_range_to_rgb(1.5,1,2,3))