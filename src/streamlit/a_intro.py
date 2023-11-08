def content(st) :
    st.title("FloraFlow : cultivez mieux avec l'IA")
    st.markdown("""
            <p class="text-justify">
            Les plantes représentent un défi complexe en termes de classification et de reconnaissance des espèces, 
            notamment en ce qui concerne les mauvaises herbes nuisibles, un enjeu pour lequel l'apprentissage automatique et 
            la vision par ordinateur offrent des solutions prometteuses pour les agriculteurs de demain.
            </p>
            """, unsafe_allow_html=True)
    st.subheader("Problématique :")
    st.markdown("""
            <p class="text-justify">
            La réussite de la culture du maïs, par exemple, dépend en grande partie de l'efficacité de la lutte 
            contre les mauvaises herbes, en particulier au cours des six à huit premières semaines après la plantation. 
            Les mauvaises herbes peuvent entraîner d'importantes pertes de rendement, allant de 10 à 100 %, en fonction de divers 
            facteurs tels que le type de mauvaises herbes et les conditions environnementales. Lutter efficacement contre 
            ces mauvaises herbes nécessite une identification précise, un processus souvent long et sujet à des erreurs.
            </p>
            """, unsafe_allow_html=True)
    st.subheader("Objectif du projet :")
    st.markdown("""
            <p class="text-justify">
            Ce projet a pour objectif de développer un modèle de deep learning capable de classer avec précision différentes espèces de plantes, en vue d'améliorer la gestion des cultures. Nous utiliserons le jeu de données Kaggle V2 Plant Seedlings Dataset, qui comprend une collection d'images de semis de plantes appartenant à diverses espèces.
            </p>
            <p class="text-justify">Les principales étapes de notre projet comprendront :</p>
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
    return st