import streamlit as st

def content() :
    col1,col2 = st.columns([0.2,0.75])
    with col1:
        st.image("src/streamlit/fichiers/FloraFlow.png", width=100)
    with col2:
        st.title("Cultivez mieux avec l'IA")

    st.image("src/streamlit/fichiers/robot.gif")
    st.header("Objectif du projet :")
    st.markdown("""
            <p class="text-justify">
            Au cours de ce projet, nous avons tenté de développer un modèle de deep learning capable de classer avec précision différentes espèces de plantes, en vue d'améliorer la gestion des cultures. Nous avons utilisé le <strong>jeu de données V2 Plant Seedlings</strong> disponible sur <a href="https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset">Kaggle</a>. C'est une base de données d'images publiques pour l'étalonnage des algorithmes de classification des semis de plantes.
            </p>
            <p class="text-justify">Ce site interactif décrit les principales étapes de ce projet :</p>
            <ul class="text-justify">
                <li>Exploration du jeu de données</li>
                <li>Conception et mise en œuvre d’un modèle d'apprentissage</li>
                <li>Interprétabilité des résultats de ce modèle</li>
                <li>Utilisation de ce modèle</li>
                <li>Prospectives</li>
            </ul>
            <p class="text-justify">
            Notre objectif final est de fournir les bases d’un outil robuste pour la classification des plantes, offrant ainsi aux agriculteurs des moyens plus efficaces de gérer leurs cultures.
            </p>
            <br>
            <p class="text-justify">
            La description exhaustive de l’étude peut être consulter dans le <a href="https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/references/Projet%20AU23_Plantes%20-%20FloraFlow.pdf">rapport associé.</a>
            </p>
            """, unsafe_allow_html=True)