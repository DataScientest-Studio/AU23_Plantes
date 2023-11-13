import streamlit as st
import pandas as pd 
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *



data = pd.read_csv("src/streamlit/fichiers/dataset_plantes.csv")
especes = list(data['Classe'].unique())
dis_classe = distribution_des_classes(data)
poids_median = poids_median_resolution(data)
data, ratios = ratios_images(data)
data, diagramme_rgba = repartition_rgb_rgba(data)
data, histogramme = repartition_especes_images_rgba(data)


def content() :
    st.title("Exploration du jeu de données")
    st.markdown("""
    <div class="text-justify">
        Le dataset V2 Plant Seedlings est composé de <strong>5539 images</strong> représentant des plants au stade de la germination. Elles sont regroupées en <strong>12 classes, représentant chacune une variété / espèce de plantes.</strong>
    </div>
    """, unsafe_allow_html=True)
    st.write("""

    """)
    with st.expander("## Exemple de photos pour chaque espèce"):
        st.image(images_especes)    
    st.write("""

    """)
    with st.expander("## Exploration du DataFrame"):
        if 'selection_especes' not in st.session_state:
            st.session_state.selection_especes = []
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            if st.button("Tout sélectionner"):
                st.session_state.selection_especes = especes
            if st.button("Effacer la sélection"):
                st.session_state.selection_especes = []
        with col2:
            selection_especes = st.multiselect("**Filtrer par espèces:**", especes,
                                               default=st.session_state.selection_especes)

        col1, col2 = st.columns(2)
        with col1:
            # Filtre sur Hauteur
            min_height, max_height = st.slider("**Filtrer par hauteur:**", min_value=int(data['Hauteur'].min()),
                                               max_value=int(data['Hauteur'].max()), value=(100, 500))
        with col2:
            # Filtre sur Largeur
            min_width, max_width = st.slider("**Filtrer par largeur:**", min_value=int(data['Largeur'].min()),
                                             max_value=int(data['Largeur'].max()), value=(100, 500))

        col1, col2 = st.columns(2)
        with col1:
            # Filtre sur Ratio
            min_ratio, max_ratio = st.slider("**Filtrer par ratio:**", min_value=float(data['Ratio'].min()),
                                             max_value=float(data['Ratio'].max()), value=(0.9, 1.15), step=0.01)
        with col2:
            # Filtre sur le nombre de canaux
            choix_canal = st.radio("**Filtrer par nombre de canaux:**", options=['3', '4', '3 et 4'], horizontal=True)

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
    col1, col2, col3, col4 = st.columns(4)

    btn1 = col1.button('Distribution des classes')
    btn2 = col2.button('Poids médian selon l\'espèce')
    btn3 = col3.button('Ratio H/L par espèce')
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

    st.header("Insight :")
    st.write("""
    - Les dimensions des images sont très variées, allant de 49x49 à 3457x3652 pixels. 
        - Très peu d’images au-delà de 1500x1500 pixels. 
        - La moyenne est d'environ 355x355 pixels 
        - La valeur médiane est de 267x267 pixels.
    - Bien que 98,77% des images aient un ratio hauteur-largeur égal à 1 (c'est-à-dire qu'elles sont carrées), certaines présentent un ratio qui s'approche de 1 ou le dépasse sans toutefois y être égal.
    - La grande majorité des images est en format RGB. Seules quelques unes (0,433%) sont en format RGBA. 
    - Le poids des images et par conséquent leur qualité est très déséquilibré
    - Il y a un déséquilibre des classes qu'il faudra prendre en compte lors de la modélisation
    - Le fonds des images n'est pas toujours uniforme (gravier / bache noire / objets divers ...)
    - Trois images ne contiennent pas de plantes 
    """)
