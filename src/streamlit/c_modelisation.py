import streamlit as st
from src.streamlit.mods.utils import get_plotly_html
import streamlit.components.v1 as html

def content():
    st.title("Modélisation")
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
             <li>de structurer nos expérimentations de façon standardisée </li>
             <li>et d’automatiser la comparaison et la génération des rapports de résultats.</li>
         </ul>
         Le code du moteur et les notebooks de démonstration sont disponibles sur <a href="https://github.com/DataScientest-Studio/AU23_Plantes/tree/main/notebooks">GitHub</a>.
        </div>
    </div>    
            """, unsafe_allow_html=True)
        st.text('Spoiler : 99% score F1 sur notre meilleur modèle')
        st.image("src/streamlit/fichiers/moteur.gif")

    with tab2:
        st.header('Choix du CNN préentrainé')
        st.markdown("""
                <div class="text-justify">
                     Nous avons essayé et comparé différents modèles CNN : quelques CNN maison et plusieurs  modèles pré-entraînés en partant des poids d’entrainement <code>imagenet</code>
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
        st.image("./src/streamlit/fichiers/CNN_precision_train.png")


    with tab3:
        st.header('Choix des paramètres étalons')
        st.markdown("""
                <div class="text-justify">
                    Nous avons défini 2 test étalons sur la base desquels la suite des expérimentations ont été réalisées  <a href="https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/src/models/final_test/stage1.py">(code sur GitHub)</a>.
                    <br><br>
                    Voici les réglages utilisés pour
                     <ul>
            <li>la séparation des données : </li> <code>80% train - 10% val - 10% test - avec classweight défini à partir des données</code>
            <br><br>
            <li>les seeds numpy et tensorflow communes à tous les tests : </li><code> np.random.seed(777), tf.random.set_seed(777)</code>
            <br><br>
            <li>les paramètres de data-augmentation :</li>
            <code> rotation_range=360,  zoom_range=0.2,  width_shift_range=0.2, height_shift_range=0.2, fill_mode="nearest"</code> 
            <br><br>
            <li>les paramètres de compilation et d'entrainement : </li>
            <code> batch_size = 32, epochs = 32, optimizer=Adam(learning_rate=1e-3),loss=CategoricalCrossentropy(), metrics=CategoricalAccuracy(), ReduceLROnPlateau('val_loss', factor=0.1, patience=3),
EarlyStopping('val_loss', patience=5), ModelCheckpoint("val_categorical_accuracy")
             </code>
             <br><br>
              <li> les couches de classification : </li>
              <code>x = self.base_model.model.output<br>
              x = Dropout(0.2)(x) <br>
                output = Dense(12, activation='softmax', name='main')(x) <br>
                self.model = Model(inputs=self.base_model.model.input, outputs=output)</code></ul>
                </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.header('Réglages')
        st.markdown("""
                <div class="text-justify">
                   Sur la base des étalons nous avons fait une série d’améliorations successives
                   <ol>
                    <li> <b>Segmentation des images</b> ≃ +4 pts F1 Accuracy (89%)</li>
                    La technique de computer vision utilisée est détaillée dans la sections suivante.  
                    <br>
                    <br>
                    <li> <b>Fine-tuning du CNN </b>≃ +6pts F1 Accuracy (95%)</li>
                    Nous réentrainons les derniers 50% des couches du CNN. 
                    Cet entrainement se fait en une seul étape, en incluant les couches de normalisation et de dropout 
                    et <b>sans passer</b> en mode inférence (<code>training=False</code>)
                    Ce choix inhabituel est basé sur un comparatif entre les différentes technique que nous avons réalisé 
                    (<a href="https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/src/testFineTuning2.py">Code</a>,
                     <a href="https://htmlpreview.github.io/?https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/reports/figures/test-finetuning2/compare_performances_accuracy.html">Illustration</a> et les explications dans le rapport)
                    <br>
                    <br>
                    <li><b>Couches de classification</b> ≃ +2pts F1 Accuracy (97%)</li>
                     Nous ajoutons le regulizer L2(1e-4) sur la couche Dense(12) en augmentant la patience à 5 de LROnPlateau (au lieu de 3 précédement). Là aussi nous avons testé de nombreuses
                      combinaisons de couche Dense (1024, 256, 128) et différent Dropout et c’est cette configuration qui nous a donné
                      les meilleurs résultats. (<a href="https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/src/testClassif2.py">Code</a>,
                     <a href="https://htmlpreview.github.io/?https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/reports/figures/test-classifpatient/compare_performances_accuracy.html">Illustration</a> et les explications dans le rapport)
                <br>
            </ol>
            Les 2 graphiques suivants 
             illustrent les performances de ces 3 tests par rapport aux étalons. 
            Le second montre en particulier que les confusions des différents modèles évoluent et se concentre particulièrement sur 2 types.
             (versions html : 
             <a href="https://htmlpreview.github.io/?https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/reports/figures/final-test2/compare_confusions.html">accuracy</a>,
         <a href="https://htmlpreview.github.io/?https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/reports/figures/final-test2/compare_confusions.html">confusion</a>)
            </div>
            """, unsafe_allow_html=True)

        st.components.v1.html(get_plotly_html("reports/figures/final-test2/compare_performances_accuracy.html"), height=600)
        st.components.v1.html(get_plotly_html("reports/figures/final-test2/compare_confusions.html"),
                              height=600)

    with tab5:
        st.header('Résultats')
        st.markdown("""
        Notre meilleur modèle est obtenu avec un CNN MobilNetV3Large, de la segmentation sémantique, du fine-tuning et de la régularisation L2 sur la couche de normalisation.
        <br>
        Les confusions du modèle sur le jeu de test se situe principalement entre les espèces Loose Silky-bent (épi du vent) et Black-grass (vulpin des champs). 
        Elles font toutes deux parties des espèces de mauvaises herbes les plus difficiles à maitriser dans les cultures.
        <br>
        Dans un contexte d’utilisation de notre modèle sur des robots desherbeurs intelligent,
         il est raisonnable de les associer comme une même espèce nuisible. (<a href="https://github.com/DataScientest-Studio/AU23_Plantes/blob/main/notebooks/3-%20Final%20Model%20results.ipynb">Code sur Github</a>)
        <br><br>
        Notre modèle devient alors très performant : 99% F1 Accuracy         
        """, unsafe_allow_html=True)
        st.image("./src/streamlit/fichiers/final_classification.png")
        st.image("./src/streamlit/fichiers/final_confusion.png")