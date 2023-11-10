
import streamlit as st
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm
import src.features as lf

def content( trainer_modeles) :
    nom_des_modeles_csv = {
        'MobileNetv3': '4-dense-Mob',
        'ResNetv2': '4-dense-Res'
    }

    nom_des_modeles_affichage = {
        'MobileNetv3': 'MobileNetv3Large',
        'ResNetv2': 'ResNetv2'
    }

    tab_titles = [nom_des_modeles_affichage.get(type(model).__name__, 'Modèle Inconnu') for model in trainer_modeles]
    tabs = st.tabs(tab_titles)

    for model, tab in zip(trainer_modeles, tabs):
        with tab:
            model_class_name = type(model).__name__
            model_name_csv = nom_des_modeles_csv.get(model_class_name, 'Inconnu')

            if model_name_csv == 'Inconnu':
                st.error(f"Le nom CSV du modèle pour la classe {model_class_name} est inconnu.")
                continue

            conf_matrix = pred_confusion_matrix(feedbacks, especes, model_name_csv)
            fig = plot_confusion_matrix(conf_matrix, especes)
            st.pyplot(fig)

    st.subheader("Conclusion")
    st.markdown("""
            <p class="text-justify">
            Les différents objectifs initiaux du projet ont été atteints :
                
            - La définition d'un modèle capable d'une grande précision, de l'ordre de 98 ± 2% suivant la méthodologie.

            - La création d'un modèle compact (34 Mo) adapté pour une intégration dans des robots aux capacités de calcul limitées.
            
            - L'établissement d'une méthode robuste, structurée en plusieurs étapes, facilite le test d'une vaste gamme d'hyper-paramètres et aide à saisir leur impact.
              Méthodologie primordiale dans une activité où la démarche empirique reste de mise. 
            
            L'implémentation de la segmentation d'images va au-delà des objectifs initialement fixés, et nous ouvre de nouvelles possibilités.
            </p>
            """, unsafe_allow_html=True)
    
    st.subheader("Prospectives")
    st.markdown("""
            <p class="text-justify">
            Nous avons plusieurs pistes d’amélioration pour continuer cette étude :

            - Le projet consiste à équiper un robot désherbeur d'une technologie capable de détecter les jeunes pousses en condition réelle.
              Cela implique la création préalable d'un dataset détaillé avec des boîtes englobantes pour chaque plante, afin de renforcer la précision et l'efficacité de la détection.
                

            </p>""", unsafe_allow_html=True)



    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("src/streamlit/fichiers/image_avec_bbox17.jpg")
    with col2:
        st.image("src/streamlit/fichiers/image_avec_bbox18.jpg")
    with col3:
        st.image("src/streamlit/fichiers/image_avec_bbox.jpg")

    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center;">Exemple avec quelques images du dataset.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
            <p class="text-justify">
                
            - Eventuellement un déploiement du modèle sur un cluster central pour un suivi et une maintenance continue.
                
            - Comparer notre modèle à des méthodes plus récentes comme YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector) et Faster R-CNN
              pour évaluer leur pertinence face à la problématique soulevé par le projet. 
             
            - L'intégration d'un mécanisme de feedback humain offre le potentiel d'accumuler un jeu de données complémentaire qui, en atteignant
            un volume conséquent, pourrait servir de base pour un apprentissage par renforcement. Ce processus permettrait d'affiner les performances du modèle
            en intégrant des nouvelles données annotées et plus variées que le jeu de données initial.
                
            </p>
            """, unsafe_allow_html=True)
