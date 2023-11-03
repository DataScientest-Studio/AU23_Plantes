# LIBRAIRIES
import time 
import requests
import os
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from io import BytesIO
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from streamlit_cropper import st_cropper
from tensorflow.keras.preprocessing import image as img_prep
from keras.models import load_model

# INITIALISATIONS
st.set_page_config(
    page_title="FloraFlow",
    page_icon="🌱"
)

# FONCTIONS
@st.cache_data
def distribution_des_classes(data):
    return px.histogram(data, x="Classe", title="Distribution des classes", color='Classe', color_discrete_sequence=color_palette)

@st.cache_data
def poids_median_resolution(data):
    df_median = data.groupby('Classe')['Résolution'].median().reset_index()
    return px.bar(df_median, x='Classe', y='Résolution', title="Résolution médiane des images selon l'espèce", color='Classe', color_discrete_sequence=color_palette)

@st.cache_data
def ratios_images(data):
    bins = [0, 0.90, 0.95, 0.99, 1.01, 1.05, max(data['Ratio']) + 0.01]
    bin_labels = ['<0.90', '0.90-0.94', '0.95-0.99', '1', '1.01-1.04', '>1.05']
    data['Ratio_cat'] = pd.cut(data['Ratio'], bins=bins, labels=bin_labels, right=False)
    count_data = data.groupby(['Ratio_cat', 'Classe']).size().reset_index(name='Nombre')
    return px.bar(count_data, x='Ratio_cat', y='Nombre', color='Classe', title="Nombre d'images par classe pour chaque catégorie de ratio", barmode='group', color_discrete_sequence=color_palette)

@st.cache_data
def repartition_rgb_rgba(data):
    compte_rgba = data[data["Canaux"] == 4].shape[0]
    compte_rgb = data[data["Canaux"] == 3].shape[0]
    valeurs = [compte_rgb, compte_rgba]
    etiquettes = ["RGB", "RGBA"]
    return px.pie(values=valeurs, names=etiquettes, title="Répartition des images en RGB et RGBA", color=etiquettes, color_discrete_sequence=color_palette)

@st.cache_data
def repartition_especes_images_rgba(data):
    donnees_rgba = data[data["Canaux"] == 4]
    repartition_especes_rgba = donnees_rgba['Classe'].value_counts().reset_index()
    repartition_especes_rgba.columns = ['Classe', 'Nombre']
    return px.bar(repartition_especes_rgba, x='Classe', y='Nombre', title='Répartition des espèces au sein des images RGBA', color='Classe', color_discrete_sequence=color_palette)

@st.cache_resource()
def load_all_models(model_names):
    models = {}
    for model_name in model_names:
        model_path = os.path.join('src/streamlit/fichiers/', model_name)
        models[model_name] = load_model(model_path)
    return models

def preprocess_image(image, target_size=(150, 150)):
        image = image.resize(target_size)
        image = image.convert("RGB")
        image_np = img_prep.img_to_array(image) / 255.0  # Rescale
        image_np = np.expand_dims(image_np, axis=0)
        return image_np

def enregistrer_feedback_pandas(url, classe_predite, bonne_classe):
    file_path = os.path.join('src', 'streamlit', 'fichiers', 'feedback', 'feedback.csv')
    df = pd.DataFrame([[url, classe_predite, bonne_classe]], columns=['URL', 'Classe Predite', 'Bonne Classe'])
    if os.path.isfile(file_path):
        df_existante = pd.read_csv(file_path)
        df_finale = pd.concat([df_existante, df], ignore_index=True)
    else:
        df_finale = df
    df_finale.to_csv(file_path, index=False)
    st.success('Feedback enregistré avec succès !')

def reset_state():
    st.session_state['feedback_soumis'] = False
    st.session_state['mauvaise_pred'] = False
    st.session_state['classe_predite'] = None
    st.session_state['resultat'] = None
    st.session_state['id_classe_predite'] = None
    st.session_state['source_image'] = None

# CSS FICTIF
color_palette = px.colors.sequential.speed
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stProgress {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)

# RESSOURCES 
# DataFrame 
data = pd.read_csv("src/streamlit/fichiers/dataset_plantes.csv")

#  Modeles  
#  Liste des modèles  
# A PERSONNALISER AVEC VOS MODELES POUR LE MOMENT ET APRES LES AVOIR INSERER DANS VOTRE DOSSIER SRC/STREAMLIT/FICHIERS LOCAL
noms_modeles = ['RESNET_CNN4-20231011-004359.h5', 'CNN6-20231007-154745.h5', 'modele_CNN1-2023-10-05_23-38-47.h5']  
# Chargement des modèles
modeles = load_all_models(noms_modeles)

# Graphs 
dis_classe = distribution_des_classes(data)
poids_median = poids_median_resolution(data)
ratios = ratios_images(data)
diagramme_rgba = repartition_rgb_rgba(data)
histogramme = repartition_especes_images_rgba(data)

# Images
images_especes = "src/streamlit/fichiers/images_especes.png"
linkedin_icon = "src/streamlit/fichiers/linkedin_icon.png"
github_icon = "src/streamlit/fichiers/github_icon.png"
ds_logo = "src/streamlit/fichiers/logo-2021.png"
logo = "src/streamlit/fichiers/FloraFlow.png"

# SIDEBAR
with st.sidebar:
    col1, col2, col3 = st.columns([0.3,0.7,0.2])
    with col2 : 
        st.image(logo)  
    choose = option_menu("", ["Introduction", "A propos du jeu de données", "DataViz", "Modélisation", "Utilisation du modèle", "Conclusion"],
                         icons=['arrow-repeat', 'database-fill', 'pie-chart', 'box-fill','cloud-download', 'check-circle-fill'],
                         default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
    footer = '''
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    # Equipe du projet
    '''

    st.sidebar.markdown(footer, unsafe_allow_html=True)


    st.sidebar.markdown("""
    #### Gilles de Peretti <a href="URL_GITHUB_GILLES" target="_blank"><i class='fab fa-github'></i></a> <a href="URL_LINKEDIN_GILLES" target="_blank"><i class='fab fa-linkedin'></i></a>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    #### Hassan ZBIB <a href="URL_GITHUB_HASSAN" target="_blank"><i class='fab fa-github'></i></a>  <a href="URL_LINKEDIN_HASSAN" target="_blank"><i class='fab fa-linkedin'></i></a>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    #### Dr. Iréné Bérenger AMIEHE ESSOMBA <a href="URL_GITHUB_IRENE" target="_blank"><i class='fab fa-github'></i></a> <a href="URL_LINKEDIN_IRENE" target="_blank"><i class='fab fa-linkedin'></i></a>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    #### Olivier MOTELET <a href="URL_GITHUB_OLIVIER" target="_blank"><i class='fab fa-github'></i></a>  <a href="URL_LINKEDIN_OLIVIER" target="_blank"><i class='fab fa-linkedin'></i></a>
    """, unsafe_allow_html=True)
    st.write("""


    """)
    st.sidebar.image(ds_logo)  

# PAGES
if choose == "Introduction":
    st.title("FloraFlow : cultivez mieux avec l'IA")
    st.write("""Les plantes représentent un défi complexe en termes de classification et de reconnaissance des espèces, 
    notamment en ce qui concerne les mauvaises herbes nuisibles, un enjeu pour lequel l'apprentissage automatique et 
    la vision par ordinateur offrent des solutions prometteuses pour les agriculteurs de demain. """)
    st.subheader("Problématique :")
    st.write("""La réussite de la culture du maïs, par exemple, dépend en grande partie de l'efficacité de la lutte 
    contre les mauvaises herbes, en particulier au cours des six à huit premières semaines après la plantation. 
    Les mauvaises herbes peuvent entraîner d'importantes pertes de rendement, allant de 10 à 100 %, en fonction de divers 
    facteurs tels que le type de mauvaises herbes et les conditions environnementales. Lutter efficacement contre 
    ces mauvaises herbes nécessite une identification précise, un processus souvent long et sujet à des erreurs.""")     
    st.subheader("Objectif du projet :")
    st.write("""
    Ce projet a pour objectif de développer un modèle de deep learning capable de classer avec précision différentes espèces de plantes, en vue d'améliorer la gestion des cultures. Nous utiliserons le jeu de données Kaggle V2 Plant Seedlings Dataset, qui comprend une collection d'images de semis de plantes appartenant à diverses espèces.

    Les principales étapes de notre projet comprendront :
    - Exploration approfondie du jeu de données
    - Préparation des données
    - Conception et mise en œuvre d'un modèle d'apprentissage
    - Analyse de ses performances

    Notre objectif final est de fournir les bases d’un outil robuste pour la classification des plantes, offrant ainsi aux agriculteurs des moyens plus efficaces de gérer leurs cultures.
    """)

if choose == "A propos du jeu de données":
    st.header("Exploration du jeu de données")
    st.write("""
    Le jeu de données utilisé est intitulé V2 Plant Seedlings Dataset disponible sur Kaggle est une base de données d'images publiques pour l'étalonnage des algorithmes de classification des semis de plantes.
    Le dataset V2 Plant Seedlings est composé de 5539 images représentant des plants au stade de la germination. Elles sont regroupées en 12 classes, représentant chacune une variété / espèce  de plantes.
    """)
    st.markdown("***Exemple de photos pour chaque espèce***")
    st.image(images_especes)
    st.markdown("""
    L'exploration de ce jeu de données comprendra les étapes suivantes :

    - Analyse des caractéristiques du jeu de données.
    - Exploration du contenu des images.
    """)


if choose == "DataViz":
    st.header("Analyse des caractéristiques du jeu de données")
    st.markdown("""
    Une fois le contenu du dataset chargé sur nos machines, nous avons créé un DataFrame afin de récupérer 
    les caractéristiques principales du jeu de donnée. Nous avons récupéré les informations essentielles et procédé à des calculs pour recueillir sur chaque image : 
    - La classe (espèce) à laquelle elle appartient
    - Le nom du fichier correspondant
    - Le chemin pour y accéder
    - Sa hauteur (H) et sa largeur (L)
    - Son ratio, défini par les deux variables précédentes
    - Sa shape (H / L / Nombre de canaux)
    - Résolution (HxL)
    """)
    st.write("""
    """)
    # Exploration intéractive du DataFrame
    with st.expander("## Exploration du DataFrame"):
        especes = list(data['Classe'].unique())

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
    st.write("### Modélisation")
    # Charger le contenu HTML depuis un fichier
    with open("src/streamlit/fichiers/test2.html", "r", encoding='utf-8') as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=600)
    

if choose == "Utilisation du modèle":
    aspect_ratio = (1, 1)
    box_color = '#e69138'

    choix_modele = st.selectbox("Choisissez un modèle", noms_modeles)
    modele = modeles[choix_modele]
    option = st.selectbox("Comment voulez-vous télécharger une image ?", ("Choisir une image de la galerie", "Télécharger une image", "Utiliser une URL"))

    image = None  
    progress_bar = st.progress(0)
    
    if option == 'Télécharger une image':
        with st.expander("ℹ️ Informations"):
            mais_1 = 'src/streamlit/fichiers/mais_1.png'
            mais_2 = 'src/streamlit/fichiers/mais_2.png'
            mauvaise_blackgrass = 'src/streamlit/fichiers/mauvaise_blackgrass.png'
            bonne_blackgrass = 'src/streamlit/fichiers/bonne_blackgrass.png'
            st.write("""
            Les formats d'images acceptés sont : **_jpg_**, **_png_** ou **_jpeg_**
            
            En choissant vous même, veillez à choisir une image qui
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
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            base_width = 350  
            w_percent = base_width / float(image.size[0])
            h_size = int(float(image.size[1]) * float(w_percent))
            resized_image = image.resize((base_width, h_size), Image.ANTIALIAS)

            col1, col2 = st.columns(2)

            with col1:
                st.write("##### Recadrez l'image chargée")
                cropped_img = st_cropper(resized_image, realtime_update=True, box_color=box_color, aspect_ratio=aspect_ratio)
            with col2:
                st.write("##### Aperçu de l'image à prédire")
                st.image(cropped_img, use_column_width=True)
                    

    elif option == 'Utiliser une URL':
        with st.expander("ℹ️ Informations"):
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

        url = st.text_input("Entrez l'URL de l'image")
        if url:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            base_width = 350  
            w_percent = base_width / float(image.size[0])
            h_size = int(float(image.size[1]) * float(w_percent))
            resized_image = image.resize((base_width, h_size), Image.ANTIALIAS)
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
        with st.expander("ℹ️ A propos des images de la galerie"):
            st.markdown(""" 
            - D'ou proviennent ces images ? 
                - Elles ont composé l'ensemble de Test, elles ont été séparées du Dataset originakl et le modèle n'a pas été entrainé sur ces images. 
                Un Random State fixe a été défini, de sorte à ce que les images du des ensemble Trainn, Val et Test soient toujours les mêmes. 
            - Pourquoi fournir ces images ? 
                - Pour montrer que le modèle fonctionne lors qu'on lui fourni des images ayant des similarités avec celles sur lesquelles il a été entrainé. 
            - Pour se repérer et comparer la classe de l'image avec la prédiction, il suffit de regarder le nom du fichier, qui n'impact pas la prédiction du modèle. 
            """)
        gallery_list = os.listdir('src/streamlit/fichiers/gallery')
        selected_image = st.selectbox("Sélectionnez une image", gallery_list)
        image_path = os.path.join('src/streamlit/fichiers/gallery', selected_image)
        image = Image.open(image_path)
        st.write("##### Aperçu de l'image à prédire")
        st.image(image, use_column_width=True)
    
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
    
    feedback_placeholder = st.empty()

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Prédiction", use_container_width=True) and image is not None:
            processed_image = preprocess_image(image)
            progress_bar = st.progress(0)
            progress_bar.progress(0.3)
            st.session_state['resultat'] = modele.predict(processed_image)
            progress_bar.progress(1.0)
            st.session_state['id_classe_predite'] = np.argmax(st.session_state['resultat'][0])
            class_mapping = {'Black-grass': 0, 'Charlock': 1, 'Cleavers': 2, 'Common Chickweed': 3, 'Common wheat': 4, 'Fat Hen': 5, 'Loose Silky-bent': 6, 'Maize': 7, 'Scentless Mayweed': 8, "Shepherd's Purse": 9, 'Small-flowered Cranesbill': 10, 'Sugar beet': 11}
            st.session_state['classe_predite'] = [name for name, id in class_mapping.items() if id == st.session_state['id_classe_predite']][0]
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
                    enregistrer_feedback_pandas(url, st.session_state['classe_predite'], st.session_state['classe_predite'])
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
                        enregistrer_feedback_pandas(url, st.session_state['classe_predite'], bonne_classe[0])
                        time.sleep(3)
                        feedback_placeholder.empty()
                        reset_state()
                    else:
                        st.error("Veuillez sélectionner une classe avant de confirmer.")


if choose == "Conclusion":
    st.write("### Conclusion")
