import streamlit as st

def show_information_block(info_type):
    if info_type == "basic_info":
        st.write("""
        Les formats d'images acceptés sont : **_jpg_**, **_png_** ou **_jpeg_**
        
        En choisissant vous-même, veillez à choisir une image qui
        est 'similaire' aux images du dataset initial et sur lesquelles nos modèles ont été formés. 
        
        Si la plante est à un stade de croissance relativement avancé, par exemple, cela ne marchera pas
        car le modèle a été entraîné sur des semis et jeunes plants.
        """)
        col1, col2 = st.columns([1,1])
        with col1:
            st.image('src/streamlit/fichiers/mauvaise_blackgrass.png', caption="🛑 Exemple d'image que le modèle n'a jamais vu")
        with col2:
            st.image('src/streamlit/fichiers/bonne_blackgrass.png', caption="✅ Image qui peut fonctionner")
        st.write("""
        
        De même, privilégiez les images prises sur le dessus et non sur la coupe, ici encore, nos modèles ayant
        été entraînés sur des images prises d'une certaine manière et sur seulement 5536 images...
        """)
        col1, col2 = st.columns([1,1])
        with col1:
            st.image('src/streamlit/fichiers/mais_1.png', caption="🛑 Exemple d'image qui perturbe le modèle")
        with col2:
            st.image('src/streamlit/fichiers/mais_2.png', caption="✅ Image qui fonctionne à 99,99%")
        
    elif info_type == "image_url":
        mais_1 = 'src/streamlit/fichiers/mais_1.png'
        mais_2 = 'src/streamlit/fichiers/mais_2.png'
        mauvaise_blackgrass = 'src/streamlit/fichiers/mauvaise_blackgrass.png'
        bonne_blackgrass = 'src/streamlit/fichiers/bonne_blackgrass.png'
        st.write("""
        L'url doit être celui d'une image et avoir la structure suivante : 
        [https://siteweb.com/mon-image.jpg](#)
        Les formats d'images acceptés sont : **_jpg_**, **_png_** ou **_jpeg_**
        
        En choissant vous même une image sur internet, veillez à choisir une image qui
        est 'similaire' aux images du dataset initial et sur lesquelles nos modèles ont été formés. 
        
        Si la plante est à un stade de croissance relativement avancé, par exemple, cela ne marchera pas
        car le modèle a été entrainé semis et jeunes plants.           
        """)
        col1, col2= st.columns([1,1])
        with col1 : 
            st.image(mauvaise_blackgrass, caption="🛑 Exemple d'image que le modèle n'a jamais vu")
        with col2 : 
            st.image(bonne_blackgrass, caption="✅ Image qui peut fonctionner")
        st.write("""

        De même, privilegiez les images prise sur le dessus et pas sur la coupe, ici encore, nos modèles ayant
        été entrainés sur des images prises d'une certaines manière et sur seulement 5536 images ... 
        
        """)
        col1, col2= st.columns([1,1])
        with col1 : 
            st.image(mais_1, caption="🛑 Exemple d'image qui perturbe le modèle")
        with col2 : 
            st.image(mais_2, caption="✅ Image qui fonctionne à 99,99%")

    elif info_type == "gallery_info":
        st.markdown(""" 
        - D'où proviennent ces images ? 
            - Elles ont composé l'ensemble de Test, elles ont été séparées du Dataset original et le modèle n'a pas été entraîné sur ces images. 
            Un Random State fixe a été défini, de sorte à ce que les images des ensembles Train, Val et Test soient toujours les mêmes. 
        - Pourquoi fournir ces images ? 
            - Pour montrer que le modèle fonctionne lorsqu'on lui fournit des images ayant des similarités avec celles sur lesquelles il a été entraîné. 
        - Pour se repérer et comparer la classe de l'image avec la prédiction, il suffit de regarder le nom du fichier, qui n'impacte pas la prédiction du modèle. 
        """)

    else:
        st.error("Type d'information non reconnu.")


def show_information_block(info_type):
    if info_type == "basic_info":
        st.markdown("""
        🌟 **L'ABC des formats acceptés :** JPG, PNG, JPEG. Gardez-les en tête !

        🔍 **Choisir avec soin :** Votre image doit être proche de celles que notre modèle connaît. Si votre plante a plus vécu que nos jeunes pousses entraînées, elle pourrait bien le dérouter !

        🖼️ **L'art du placement :** Vue du dessus = 🎯. Vue de côté = 🚫. Nos modèles sont de véritables artistes de la vue aérienne !
        """)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image('src/streamlit/fichiers/mauvaise_blackgrass.png', caption="🛑 Pas comme ça")
        with col2:
            st.image('src/streamlit/fichiers/bonne_blackgrass.png', caption="✅ C'est ça !")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image('src/streamlit/fichiers/mais_2.png', caption="✅ C'est ça !")
        with col2:
            st.image('src/streamlit/fichiers/mais_1.png', caption="🛑 Pas comme ça")
        
    elif info_type == "image_url":
        st.markdown("""
        🔗 **L'Adresse de la réussite :** L'url de votre image devrait ressembler à [https://siteweb.com/mon-image.jpg](#). Assurez-vous qu'elle ressemble aux images d'entraînement pour une prédiction en or !

        🌟 **L'ABC des formats acceptés :** JPG, PNG, JPEG. Gardez-les en tête !

        🔍 **Choisir avec soin :** Votre image doit être proche de celles que notre modèle connaît. Si votre plante a plus vécu que nos jeunes pousses entraînées, elle pourrait bien le dérouter !

        🖼️ **L'art du placement :** Vue du dessus = 🎯. Vue de côté = 🚫. Nos modèles sont de véritables artistes de la vue aérienne !
        """)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image('src/streamlit/fichiers/mauvaise_blackgrass.png', caption="🛑 Pas comme ça")
        with col2:
            st.image('src/streamlit/fichiers/bonne_blackgrass.png', caption="✅ C'est ça !")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image('src/streamlit/fichiers/mais_2.png', caption="✅ C'est ça !")
        with col2:
            st.image('src/streamlit/fichiers/mais_1.png', caption="🛑 Pas comme ça")
        
    elif info_type == "gallery_info":
        st.markdown(""" 
        🏞️ **Galerie de l'excellence :**
        - 📸 **Origine des oeuvres :** Ces images n'ont pas posé pour notre modèle. Elles sont vierges de tout entraînement, choisies par un Random State artistique, pour autant elles ont des caractéristiques similaires à celles du jeu d'entraînement.
        - 🌟 **Raison d'être :** Elles démontrent que notre modèle est un virtuose lorsqu'il s'agit d'images familières.
        - 🔍 **Nom de fichier = Clé de voûte :** Pour comparer classe et prédiction, le nom du fichier est votre guide sans influencer notre modèle.
        """)

    else:
        st.error("Erreur dans le choix de l'info type")
