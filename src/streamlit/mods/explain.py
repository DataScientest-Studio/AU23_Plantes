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


code_CNN = """
Conv2D(32, (6, 6), activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(150, 150, 3)),
BatchNormalization(),
MaxPooling2D(2, 2),

Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
BatchNormalization(),
MaxPooling2D(2, 2),

Conv2D(128, (4, 4), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
BatchNormalization(),
MaxPooling2D(2, 2),

Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
BatchNormalization(),
MaxPooling2D(2, 2),

Flatten(),

Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
Dropout(0.5),

Dense(12, activation='softmax')
"""


