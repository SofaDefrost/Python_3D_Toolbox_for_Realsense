import pyvista as pv

# Cette fonction réduit le nombre de point d'un .ply selon une certaine densité (par exemple densite=0.5 signifie que vous conservez 50% des points)
def reduction_densite_pc(input_file, output_file, densite):

    # Charger le nuage de points
    cloud = pv.read(input_file)

    # Réduire la densité de points en utilisant la méthode "decimate"
    cloud_reduced = cloud.decimate(densite)  

    # Sauvegarder le nuage de points réduit
    cloud_reduced.save(output_file)

    print(f"Nuage de points réduit enregistré sous : {output_file}")
