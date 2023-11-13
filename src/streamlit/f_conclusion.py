
import streamlit as st
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm
import src.features as lf

def content() :
    st.title("Conclusion")
    st.markdown("""
            <p class="text-justify">
            Les différents objectifs initiaux du projet ont été atteints :
                
            - La définition d'un <strong>modèle</strong> capable d'une <strong>grande précision</strong>, de l'ordre de <strong>98 ± 2%</strong> suivant la méthodologie.
            - La création d'un <strong>modèle compact (34 Mo)</strong> adapté pour une intégration dans des robots aux capacités de calcul limitées.
            
            - L'établissement d'une <strong>méthode robuste</strong>, structurée en plusieurs étapes, facilite le test d'une vaste gamme d'hyper-paramètres et aide à saisir leur impact.
              Méthodologie primordiale dans une activité où la démarche empirique reste de mise. 
            
            L'implémentation de la <strong>segmentation d'images</strong> va au-delà des objectifs initialement fixés, et nous ouvre de nouvelles possibilités.
            </p>
            """, unsafe_allow_html=True)
    
    st.header("Prospectives")
    st.markdown("""
            <p class="text-justify">
            Nous avons plusieurs <strong>pistes d’amélioration</strong> pour continuer cette étude :

            - Le projet consiste à <strong>équiper un robot désherbeur</strong> d'une technologie capable de détecter les jeunes pousses en <strong>condition réelle</strong>.
              Cela implique la création préalable d'un dataset détaillé avec des <strong>boîtes englobantes</strong> pour chaque plante, afin de renforcer la précision et l'efficacité de la détection.
                
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
                
            - Eventuellement un <strong>déploiement</strong> du modèle sur un <strong>cluster central pour un suivi et une maintenance continue.
                
            - Explorer d’autres <strong>méthodes plus récentes</strong> ou complémentaire comme YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector), Faster R-CNN ou les Vision Transformers (ViT)  pour évaluer leur pertinence face à la problématique soulevé par le projet. 
             
            - L'intégration d'un mécanisme de <strong>feedback humain</strong> offre le potentiel d'accumuler un jeu de données complémentaire qui, en atteignant
            un volume conséquent, pourrait servir de base pour un <strong>apprentissage par renforcement</strong>. Ce processus permettrait d'affiner les performances du modèle
            en intégrant des nouvelles données annotées et plus variées que le jeu de données initial.
                
            </p>
            """, unsafe_allow_html=True)