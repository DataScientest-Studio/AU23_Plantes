import streamlit as st

def content() :
    st.title("FloraFlow : cultivez mieux avec l'IA")
    st.image("src/streamlit/fichiers/robot.gif")
    st.subheader("Objectif du projet :")
    st.markdown("""
            <p class="text-justify">
            Au cours de ce projet, nous avons tenté de développer un modèle de deep learning capable de classer avec précision différentes espèces de plantes, en vue d'améliorer la gestion des cultures. Nous avons utilisé le <a href="https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset">V2 Plant Seedlings Dataset disponible sur Kaggle</a>. C'est une base de données d'images publiques pour l'étalonnage des algorithmes de classification des semis de plantes.
            </p>
            <p class="text-justify">Les principales étapes de notre projet ont été :</p>
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