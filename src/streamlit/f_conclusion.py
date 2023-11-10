
import streamlit as st
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm
import src.features as lf

def content() :
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
                
            - Eventuellement déploiement du modèle sur un cluster central pour un suivi et une maintenance continue.
                
            - Comparer notre modèles à des méthodes plus récentes comme YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector) et Faster R-CNN
              pour évaluer leur pertinence face à la problématique soulevé par le projet. 
             
            - L'intégration d'un mécanisme de feedback humain offre le potentiel d'accumuler un jeu de données complémentaire qui, en atteignant
            un volume conséquent, pourrait servir de base pour un apprentissage par renforcement. Ce processus permettrait d'affiner les performances du modèle
            en intégrant des nouvelles données annotées et plus variées que le jeu de données initial.
                
            </p>
            """, unsafe_allow_html=True)