# LIBRAIRIES
import time 
import requests
import os
import numpy as np
import streamlit as st
from streamlit_cropper import st_cropper
from src.streamlit.mods.histogram_color import compute 
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm

st.set_page_config(**page_config)
load_css('src/streamlit/mods/styles.css')

lm.models.RECORD_DIR='./models/records'
lm.models.FIGURE_DIR='./reports/figures'



trainer_modeles = load_all_models()

# Graphs 
dis_classe = distribution_des_classes(data)
poids_median = poids_median_resolution(data)
ratios = ratios_images(data)
diagramme_rgba = repartition_rgb_rgba(data)
histogramme = repartition_especes_images_rgba(data)

choose = setup_sidebar()
# PAGES

if choose == "Introduction":
    st.title("FloraFlow : cultivez mieux avec l'IA")
    st.markdown("""
        <p class="text-justify">
        Les plantes représentent un défi complexe en termes de classification et de reconnaissance des espèces, 
        notamment en ce qui concerne les mauvaises herbes nuisibles, un enjeu pour lequel l'apprentissage automatique et 
        la vision par ordinateur offrent des solutions prometteuses pour les agriculteurs de demain.
        </p>
        """, unsafe_allow_html=True)
    st.subheader("Problématique :")
    st.markdown("""
        <p class="text-justify">
        La réussite de la culture du maïs, par exemple, dépend en grande partie de l'efficacité de la lutte 
        contre les mauvaises herbes, en particulier au cours des six à huit premières semaines après la plantation. 
        Les mauvaises herbes peuvent entraîner d'importantes pertes de rendement, allant de 10 à 100 %, en fonction de divers 
        facteurs tels que le type de mauvaises herbes et les conditions environnementales. Lutter efficacement contre 
        ces mauvaises herbes nécessite une identification précise, un processus souvent long et sujet à des erreurs.
        </p>
        """, unsafe_allow_html=True)     
    st.subheader("Objectif du projet :")
    st.markdown("""
        <p class="text-justify">
        Ce projet a pour objectif de développer un modèle de deep learning capable de classer avec précision différentes espèces de plantes, en vue d'améliorer la gestion des cultures. Nous utiliserons le jeu de données Kaggle V2 Plant Seedlings Dataset, qui comprend une collection d'images de semis de plantes appartenant à diverses espèces.
        </p>
        <p class="text-justify">Les principales étapes de notre projet comprendront :</p>
        <ul class="text-justify">
            <li>Exploration approfondie du jeu de données</li>
            <li>Préparation des données</li>
            <li>Conception et mise en œuvre d'un modèle d'apprentissage</li>
            <li>Analyse de ses performances</li>
        </ul>
        <p class="text-justify">
        Notre objectif final est de fournir les bases d’un outil robuste pour la classification des plantes, offrant ainsi aux agriculteurs des moyens plus efficaces de gérer leurs cultures.
        </p>
        """, unsafe_allow_html=True)

if choose == "A propos du jeu de données":
    st.header("Exploration du jeu de données")
    st.markdown("""
    <div class="text-justify">
        Le jeu de données utilisé est intitulé V2 Plant Seedlings Dataset disponible sur Kaggle est une base de données d'images publiques pour l'étalonnage des algorithmes de classification des semis de plantes.
        Le dataset V2 Plant Seedlings est composé de 5539 images représentant des plants au stade de la germination. Elles sont regroupées en 12 classes, représentant chacune une variété / espèce de plantes.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("***Exemple de photos pour chaque espèce***", unsafe_allow_html=True)
    st.image(images_especes)
    st.markdown("""
    <div class="text-justify">
        L'exploration de ce jeu de données comprendra les étapes suivantes :

        - Analyse des caractéristiques du jeu de données.
        - Exploration du contenu des images.
    </div>
    """, unsafe_allow_html=True)

if choose == "DataViz":
    st.header("Analyse des caractéristiques du jeu de données")
    st.markdown("""
    <div class="text-justify">
        Une fois le contenu du dataset chargé sur nos machines, nous avons créé un DataFrame afin de récupérer 
        les caractéristiques principales du jeu de donnée. Nous avons récupéré les informations essentielles et procédé à des calculs pour recueillir sur chaque image : 
        <ul>
            <li>La classe (espèce) à laquelle elle appartient</li>
            <li>Le nom du fichier correspondant</li>
            <li>Le chemin pour y accéder</li>
            <li>Sa hauteur (H) et sa largeur (L)</li>
            <li>Son ratio, défini par les deux variables précédentes</li>
            <li>Sa shape (H / L / Nombre de canaux)</li>
            <li>Résolution (HxL)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Exploration intéractive du DataFrame
    with st.expander("## Exploration du DataFrame"):
        # Filtrage des espèces 
        if 'selection_especes' not in st.session_state:
            st.session_state.selection_especes = []
        col1, col2 = st.columns([0.3, 0.7])
        with col1: 
            if st.button("Tout sélectionner"):
                st.session_state.selection_especes = especes
            if st.button("Effacer la sélection"):
                st.session_state.selection_especes = []
        with col2: 
            selection_especes = st.multiselect("**Filtrez par espèces:**", especes, default=st.session_state.selection_especes)

        col1, col2 = st.columns(2)
        with col1: 
            # Filtre sur Hauteur
            min_height, max_height = st.slider("**Filtrez par hauteur:**", min_value=int(data['Hauteur'].min()), max_value=int(data['Hauteur'].max()), value=(100, 500))
        with col2: 
            # Filtre sur Largeur
            min_width, max_width = st.slider("**Filtrez par largeur:**", min_value=int(data['Largeur'].min()), max_value=int(data['Largeur'].max()), value=(100, 500))
        
        col1, col2 = st.columns(2)
        with col1:  
            # Filtre sur Ratio
            min_ratio, max_ratio = st.slider("**Filtrez par ratio:**", min_value=float(data['Ratio'].min()), max_value=float(data['Ratio'].max()), value=(0.9, 1.15), step=0.01)
        with col2: 
            # Filtre sur le nombre de canaux
            choix_canal = st.radio("**Filtrez par nombre de canaux:**", options=['3', '4', '3 et 4'], horizontal=True)
        
        # Filtrer et afficher le dataframe
        data_filtre = data[
            data['Classe'].isin(selection_especes) &
            (data['Hauteur'] >= min_height) & (data['Hauteur'] <= max_height) &
            (data['Largeur'] >= min_width) & (data['Largeur'] <= max_width) &
            (data['Ratio'] >= min_ratio) & (data['Ratio'] <= max_ratio)]
        
        if choix_canal == '3':
            data_filtre = data_filtre[data_filtre['Canaux'] == 3]
        elif choix_canal == '4':
            data_filtre = data_filtre[data_filtre['Canaux'] == 4]
        elif choix_canal == '3 et 4':
            data_filtre = data_filtre[data_filtre['Canaux'].isin([3, 4])]
        
        st.dataframe(data_filtre)


    st.write("""


    """)
    st.subheader("Visualisation des données ⬇️ ")
    st.write("""


    """)
    col1, col2, col3, col4 = st.columns(4)

    btn1 = col1.button('Distribution des classes')
    btn2 = col2.button('Poids médian selon l\'espèce')
    btn3 = col3.button('Rapport Hauteur/Largeur par espèce')
    btn4 = col4.button('Répartion des images en RGBA')


    # Barre bouton pour DataViz
    if btn1:
        st.plotly_chart(dis_classe, use_container_width=True)
    if btn2:
        st.plotly_chart(poids_median, use_container_width=True)

    if btn3:
        st.plotly_chart(ratios, use_container_width=True)
    
    if btn4:
        onglet1, onglet2 = st.tabs(["Répartition RGB/RGBA", "Répartition des espèces en RGBA"])
        with onglet1:
            st.plotly_chart(diagramme_rgba, use_container_width=True)
        with onglet2:
            st.plotly_chart(histogramme, use_container_width=True)

    st.write("##### Insight que nous pouvons retirer :")
    st.write("""
    - Les dimensions des images sont très variées, allant de 49x49 à 3457x3652 pixels. 
        - Très peu d’images au-delà de 1500x1500 pixels. 
        - La moyenne est d'environ 355x355 pixels 
        - La valeur médiane est de 267x267 pixels.
    - Bien que 98,77% des images aient un ratio hauteur-largeur égal à 1 (c'est-à-dire qu'elles sont carrées), certaines présentent un ratio qui s'approche de 1 ou le dépasse sans toutefois y être égal.
    - La grande majorité des images est en format RGB. Seules quelques unes (0,433%) sont en format RGBA. 
    - Il y a un déséquilibre des classes qu'il faudra prendre en compte lors de la modélisation
    """)
    
if choose == "Modélisation":
    st.header("Approche de modélisation")
    st.markdown("""
    <div class="text-justify">
        Étant donné nos capacités de traitement des données (machines personnelles), nous nous sommes concentrés sur des modèles de type CNN. Cette phase de modélisation comprend une série d’étapes et de choix :
        <ul>
            <li>Préparation des données,</li>
            <li>Choix du CNN,</li>
            <li>Réglage fin,</li>
            <li>Résultats finaux et pistes d’amélioration.</li>
        </ul>
        Ces choix se baseront non seulement sur les résultats de nos expérimentations mais aussi sur :
        <ul>
            <li>Notre contexte de développement – nos machines sont relativement peu puissantes et ne peuvent entraîner des modèles très gourmands,</li>
            <li>Le contexte d’usage – il est raisonnable d’imaginer que les robots de désherbage automatiques pouvant appliquer ce genre de modèle seront des machines avec des ressources limitées.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <blockquote>
    I have also learned not to take glory in the difficulty of a proof: difficulty means we have not understood.
    The ideal is to be able to paint a landscape in which the proof is obvious.
    <cite>— Pierre Deligne, Notices of the American Mathematical Society, Volume 63, Number 3, pp. 250, March 2016</cite>
    </blockquote>
    """, unsafe_allow_html=True)

    st.subheader('Préparation des données')
    tab1, tab2, tab3 = st.tabs(["Data Augmentation", "Création du jeu de test", "Redimensionnement des images"])
    with tab1:
        st.markdown("""
        <div class="text-justify">
            Notre phase d’exploration nous a montré que nous avons un jeu de données réduit et déséquilibré. Nous traiterons la partie équilibrage dans un second temps. Dans un premier temps, nous nous concentrons sur l’augmentation des données. Pour se faire, nous appliquerons les filtres suivants :
            <ul>
                <li>Rotations aléatoires: <code>rotation_range=360</code>,</li>
                <li>Zooms aléatoires: <code>zoom_range=0.2</code>,</li>
                <li>Translations aléatoires: <code>width_shift_range=0.2</code>, <code>height_shift_range=0.2</code>,</li>
                <li>Remplissage des parties manquantes: <code>fill_mode="nearest"</code></li>
            </ul>
            Ces choix ont été faits pour générer le plus de diversité possible. Nous avons aussi essayé d’utiliser un cisaillement (shear) mais nos tests ont montré que cette déformation avait un impact négatif sur nos résultats. Nous présumons que c’est parce que la forme des feuilles est importante dans la reconnaissance d’une plante.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="text-justify">
            Dans un premier temps nous avons utilisé la fonction <code>image_dataset_from_directory</code> de keras, directe et amenant de bonnes performances. 
            Mais nous avons finalement utilisé un <code>ImageDataGenerator</code> avec la fonction <code>flow_from_dataframe</code> afin de garder plus de contrôle sur nos données de tests et pouvoir retester nos modèles plus facilement en association avec une random seed constante. 
            Afin de garder un équilibre entre la quantité de données d'entraînement et un jeu de test statistiquement raisonnable, nous avons choisi le découpage suivant :
            <ul>
                <li>10% pour le jeu de test (soit 553 images),</li>
                <li>90% pour le jeu d'entraînement incluant une part de validation de 11%.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="text-justify">
            Nous avons choisi 2 tailles d’entrée dans nos modèles pendant la phase d’expérimentation toutes deux multiples de 32 :
            <ul>
                <li>128x128 : pour la légèreté que nous permet ce format réduit,</li>
                <li>224x224 : car cela se rapproche de la taille médiane (267x267) de notre dataset.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.subheader('Choix des modèles')
    st.markdown("""
        <div class="text-justify">
            Au cours de nos expérimentations individuelles, nous avons essayé différents modèles. Du CNN basique à 
            des modèles pré-entraînés. Bien qu’un de nos CNN maison ait obtenu une précision de 91%, les modèles pré-entraînés dépassent 
            largement le CNN maison.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('#### A propos du CNN maison')
    tab1, tab2, tab3, tab4 = st.tabs(["Structure du CNN", "Courbes entraînement", "Rapport de classification", "Matrice de confusion"])
    with tab1:
        st.code(code_CNN, language='python')
    with tab2: 
        courbes = 'src/streamlit/fichiers/courbe CNN4 .png'
        st.image(courbes)
    with tab3: 
        rapport = 'src/streamlit/fichiers/rapport.png'
        st.image(rapport)
    with tab4: 
        matrice = 'src/streamlit/fichiers/matrice CNN4.png'
        st.image(matrice)
    st.subheader('Comparaison des différents modèles')
    col1, col2 = st.columns(2)
    with col1:
        precision_train = 'src/streamlit/fichiers/precision_train.png' 
        st.image(precision_train)
    with col2: 
        precision_val = 'src/streamlit/fichiers/precision_val.png'
        st.image(precision_val)
    st.markdown("""
        <div class="text-justify">
        Pour la suite de cette étude, nous nous concentrerons donc sur les deux modèles : <strong>ResNet50v2</strong> et <strong>MobileNetV3Large</strong>. Néanmoins il est important de noter que : 
                <ul>
                <li> les temps d'entraînement sont bien différents entre ces 2 modèles : x2.5 (<em>62 min pour ResNet50v2 vs. 25 min sur MobileNetV3Large sur un Mac M1</em>).
                <li> La taille de stockage de ces 2 modèles est aussi très différente avec un rapport de x5 (<em> ResNet50V2 ≃ 100 à 150 MB / MobileNetV3Large ≃ 20 à 30 MB </em>)
                <li> Notons aussi que l’utilisation de MobileNetV3Large nous oblige à utiliser une taille d’image de 224x224.
                </ul>
        </div>
        """, unsafe_allow_html=True)
    st.subheader('Segmentation sémantique des images')
    compute(st=st)

if choose == "Utilisation du modèle":
    if 'classe_predite' not in st.session_state:
        st.session_state['classe_predite'] = None
    if 'resultat' not in st.session_state:
        st.session_state['resultat'] = None
    if 'id_classe_predite' not in st.session_state:
        st.session_state['id_classe_predite'] = None
    if "mauvaise_pred" not in st.session_state:
        st.session_state['mauvaise_pred'] = False
    if 'feedback_soumis' not in st.session_state:
        st.session_state['feedback_soumis'] = False
    
    aspect_ratio = (1, 1)
    box_color = '#e69138'
    choix_idx = st.selectbox("Choisissez un modèle", [0,1])
    choix_modele = trainer_modeles[choix_idx]
    option = st.selectbox("Comment voulez-vous télécharger une image ?", ("Choisir une image de la galerie", "Télécharger une image", "Utiliser une URL"))
    image = None  
    progress_bar = st.progress(0)
    if option == 'Télécharger une image':
        with st.expander("ℹ️ Informations"):
            show_information_block("basic_info")
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            base_width = 350  
            w_percent = base_width / float(image.size[0])
            h_size = int(float(image.size[1]) * float(w_percent))
            resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Recadrez l'image chargée")
                cropped_img = st_cropper(resized_image, realtime_update=True, box_color=box_color, aspect_ratio=aspect_ratio)
            with col2:
                st.write("##### Aperçu de l'image à prédire")
                st.image(cropped_img, use_column_width=True)
    elif option == 'Utiliser une URL':
        with st.expander("ℹ️ Informations"):
            show_information_block("image_url")
        url = st.text_input("Entrez l'URL de l'image")
        if url:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            base_width = 350  
            w_percent = base_width / float(image.size[0])
            h_size = int(float(image.size[1]) * float(w_percent))
            resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
            st.session_state['source_image'] = 'url'
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Recadrez l'image chargée")
                cropped_img = st_cropper(resized_image, realtime_update=True, box_color=box_color, aspect_ratio=aspect_ratio)
            with col2:
                st.write("##### Aperçu de l'image à prédire")
                st.image(cropped_img, use_column_width=True)
            image = cropped_img
    elif option == 'Choisir une image de la galerie':
        with st.expander("ℹ️ Informations"):
            show_information_block("gallery_info")
        gallery_list = os.listdir('src/streamlit/fichiers/gallery')
        selected_image = st.selectbox("Sélectionnez une image", gallery_list)
        image_path = os.path.join('src/streamlit/fichiers/gallery', selected_image)
        image = Image.open(image_path)
        st.write("##### Aperçu de l'image à prédire")
        st.image(image, use_column_width=True)
    feedback_placeholder = st.empty()
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Prédiction", use_container_width=True) and image is not None:
            #processed_image = preprocess_image(image)
            progress_bar = st.progress(0)
            progress_bar.progress(0.3)
            # st.session_state['resultat'] = choix_modele.predict_image(image)
            # progress_bar.progress(1.0)
            # st.session_state['id_classe_predite'] = np.argmax(st.session_state['resultat'][0])
            # class_mapping = {'Black-grass': 0, 'Charlock': 1, 'Cleavers': 2, 'Common Chickweed': 3, 'Common wheat': 4, 'Fat Hen': 5, 'Loose Silky-bent': 6, 'Maize': 7, 'Scentless Mayweed': 8, "Shepherd's Purse": 9, 'Small-flowered Cranesbill': 10, 'Sugar beet': 11}
            # st.session_state['classe_predite'] = [name for name, id in class_mapping.items() if id == st.session_state['id_classe_predite']][0]
            image = image.resize((224,224))
            st.session_state['classe_predite'] = choix_modele.predict_image(image)
            time.sleep(3)  
            progress_bar.progress(0)
    if (st.session_state.get('source_image') != 'url'):
        if (st.session_state['classe_predite'] is not None and st.session_state['resultat'] is not None): 
            with feedback_placeholder.container():
                st.markdown(f"Selon le modèle, il s'agit de l'espèce **{st.session_state['classe_predite']}** avec une précision de : **{st.session_state['resultat'][0][st.session_state['id_classe_predite']]*100:.2f}%**")
                reset_state()
    if (st.session_state['classe_predite'] is not None and st.session_state['resultat'] is not None and not st.session_state['feedback_soumis'] and st.session_state.get('source_image') == 'url'):
        with feedback_placeholder.container():
            st.markdown(f"Selon le modèle, il s'agit de l'espèce **{st.session_state['classe_predite']}** avec une précision de : **{st.session_state['resultat'][0][st.session_state['id_classe_predite']]*100:.2f}%**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Correcte"):
                    enregistrer_feedback_pandas(url, st.session_state['classe_predite'], st.session_state['classe_predite'], nom_modele)
                    st.session_state['feedback_soumis'] = False
                    st.session_state['classe_predite'] = None
                    st.session_state['resultat'] = None
                    st.session_state['id_classe_predite'] = None
            with col2:
                if st.button("Incorrecte"):
                    st.session_state['mauvaise_pred'] = True
            if st.session_state['mauvaise_pred']:
                especes = list(data['Classe'].unique())
                bonne_classe = st.multiselect("Précisez la bonne classe :", especes)
                if st.button('Confirmer la classe'):
                    if bonne_classe:
                        enregistrer_feedback_pandas(url, st.session_state['classe_predite'], bonne_classe[0], nom_modele)
                        time.sleep(3)
                        feedback_placeholder.empty()
                        reset_state()
                    else:
                        st.error("Veuillez sélectionner une classe avant de confirmer.")
if choose == "Conclusion":
    st.write("### Conclusion")
    tab_titles = [f"Modèle {noms_modeles[i]}" for i in range(len(noms_modeles))]
    tabs = st.tabs(tab_titles)
    for i, tab in enumerate(tabs):
        with tab:
            model_name = noms_modeles[i]
            conf_matrix = pred_confusion_matrix(feedbacks, especes, model_name)
            fig = plot_confusion_matrix(conf_matrix, especes)
            st.pyplot(fig)
