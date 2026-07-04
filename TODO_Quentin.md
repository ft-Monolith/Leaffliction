Quentin:

    - Augmentation.py (le plus important pour moi)

        Faire en sorte que Augmentation.py augmente directement les images dans le dossier d'origine, pour equilibrer les classes. (ex: si une classe a 100 images et une autre 50, augmenter la deuxième pour qu'elle ait 100 images) 

        si l argument est un dossier, augmenter les images de ce dossier pour équilibrer les classes.
        si l argument est une image, plot comme sur le sujet

        "You will need to return this data set in an "augmented_directory" for evaluation."

        Ca on le fera juste avant l eval pas besoin de se prendre la tete avec ca maintenant

    - Transformation.pY

        Juste litteralement ce que le sujet dit 
        si arg = image.jpg 
            plot l image et la version transformée (avec les transformations du sujet)
        si arg = dossier 
            tout save dans un dossier "transformed_directory" 
            et avec la possibilite de choisir le dossier de sortie

    - setup.py
        deplacer tes fichier dans src
        ensuite ajoute tes fonction et arg dans setup.py
        c'est pour que ce soit plus clean comme ca y a juste setup.py a la racine a lancer pour faire tout ce qu on veut


Felix :

    - Ajouter un clean a setup.py
    - Ajouter un auto mdoe a setup.py 
    