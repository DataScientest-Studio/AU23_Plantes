

def content(st):
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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Structure du CNN", "Courbes entraînement", "Rapport de classification", "Matrice de confusion"])
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

    plotly1 = "src/streamlit/fichiers/test2.html"
    st.components.v1.html(get_plotly_html(plotly1), height=600)
    return st