import streamlit as st
from src.streamlit.mods.utils import get_plotly_html

def content():
    st.header("Mdélisation")
    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Méthode", "CNN", "Étalons", "Réglages", "Résultats"])

    with tab1:
        st.subheader('Méthodologie')
        st.markdown("""
           <div class="text-justify">
        Lors de cette étude, nous avons rencontrés de nombreux problèmes :
          <ul>
            <li>limitation des machines,</li>
            <li>hétérogénité des méthodes utilisées entre les membres de l’équipe,</li>
              <li> difficulté à reproduire les résultats sur d'autres machines,</li>
              <li> difficulté à appréhender l’impact de nos hyperparamètres sur les performances</li>
          </ul>
    
       Pour y faire face, nous avons créé un moteur commun permettant 
         <ul>
             <li>de structurer nos expérimentations de façon standardiser </li>
             <li>et d’automatiser la génération des rapports de résultats.</li>
         </ul>
         Le code du moteur et les notebooks de démonstration sont disponibles sur <a href="https://github.com/DataScientest-Studio/AU23_Plantes/tree/main/notebooks">GitHub</a>.
    </div>    
            """, unsafe_allow_html=True)
        st.image("src/streamlit/fichiers/moteur.gif")

    with tab2:
        st.subheader('Choix du CNN préentrainé')
        st.markdown("""
                <div class="text-justify">
                     Nous avons essayé et comparé différents modèles CNN : quelques CNN maison et plusieurs  modèles pré-entraînés.
                     <br>Il ressort 2 CNN particulierement performants sur notre jeu de données : <b>ResNet</b> et <b>MobilNet</b>. 
                     <br>C’est ces deux modèles que nous utiliserons pour la suite de l’étude. 
                     A noter que ResNet est plus gourmand en ressources (5x plus d'espace disque et 2.5x plus de temps d'entrainement)
                 </div>
            """, unsafe_allow_html=True)
        st.markdown("""
                        <br><div class="text-justify">
                            Résultats sur la validation des différents CNN : 
                         </div>
                    """, unsafe_allow_html=True)
        st.image("src/streamlit/fichiers/CNN_precision_val.png")
        st.markdown("""
                        <br><div class="text-justify">
                            Résultats pendant l'entrainement des différents CNN : 
                         </div>
                    """, unsafe_allow_html=True)
        st.image("src/streamlit/fichiers/CNN_precision_train.png")

