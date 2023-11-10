# FloraFlow - Cultivez mieux avec l'IA

Ce dépôt GitHub contient tout le code nécessaire au bon fonctionnement du projet que l'on a nommé FloraFlow. Ce projet fil-rouge a été mené lors de notre formation Data Scientist chez Data Scientest. 

## Vue d'ensemble du projet :
Ce projet a pour objectif de développer un modèle de deep learning capable de classer avec précision différentes espèces de plantes, en vue d'améliorer la gestion des cultures. 

Les principales étapes de notre projet comprennent :  
- l'exploration approfondie du jeu de données, 
- la préparation des données, 
- la conception et la mise en œuvre d'un modèle d'apprentissage, 
- ainsi que l'analyse de ses performances.

Notre objectif final est de fournir les bases d’un outil robuste pour la classification des plantes, offrant ainsi aux agriculteurs des moyens plus efficaces de gérer leurs cultures.

Derrière ce FloraFlow se trouvent 4 personnes en formation chez DataScientest:

- Gilles de Peretti [Linkedin](https://www.linkedin.com/in/gilles-de-peretti-8219425a/) - [GitHub](https://github.com/gillesdeperetti)
- Hassan ZBIB [Linkedin](#) - [GitHub](#)
- Dr. Iréné AMIEHE ESSOMBA [Linkedin](https://www.linkedin.com/in/amiehe-essomba "Amiehe Essomba") - [GitHub](https://github.com/amiehe-essomba "Amiehe Essomba")
- Olivier MOTELET [Linkedin](#) - [GitHub](#)

## A propos du dataset 

Nous avons utilisé le jeu de données Kaggle [V2 Plant Seedlings Dataset](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset), qui comprend une collection d'images de plantes appartenant à diverses espèces. 

Le dataset V2 Plant Seedlings est composé de 5539 images représentant des plants au stade de la germination. Elles sont regroupées en 12 classes, représentant chacune une variété / espèce  de plantes.

## Structure du projet
------------

    ├── LICENSE
    ├── README.md
    ├── models
    ├── notebooks
    │   ├── 0- Data exploration.ipynb
    │   ├── 1- Trainer demo.ipynb
    │   ├── 2- Campaign demo.ipynb
    │   └── 3- Model results.ipynb
    ├── packages.txt
    ├── references
    ├── reports
    │   └── figures
    ├── requirements.txt
    └── src
        ├── __init__.py
        ├── features
        │   ├── __init__.py
        │   ├── data_builder.py
        │   └── segmentation.py
        ├── main.py
        ├── models
        │   ├── __init__.py
        │   ├── campaign.py
        │   ├── final_test
        │   │   ├── __init__.py
        │   │   ├── stage1.py
        │   │   ├── stage2.py
        │   │   ├── stage3.py
        │   │   └── stage4.py
        │   ├── model_wrapper.py
        │   ├── models.py
        ├── streamlit
        │   ├── app.py
        │   ├── config.toml
        |   ├── mods
        |   |    ├── __init__.py
        |   |    └── histogram_color.py
        │   └── fichiers
        │       └── gallery
        └── visualization
            ├── __init__.py
            ├── eda.py
            └── graphs.py

--------

## Installation

Pour commencer à utiliser FloraFlow, clonez le dépôt et installez les dépendances nécessaires :

    ```bash
    git clone https://github.com/DataScientest-Studio/AU23_Plantes.git FloraFlow
    cd FloraFlow
    pip install -r requirements.txt
    ```

Ensuite téléchargez le jeu de données de Kaggle cité plus haut et placez le dans votre dossier local FloraFlow. Il est conseillé de ne pas utiliser le dossier NonSegmented qui peut contenir plusieur plants de différentes espèces au sein d'une même image. 

## Utilisation

Pour lancer l'application Streamlit et interagir avec l'interface utilisateur :

```bash
streamlit run streamlit/app.py 
```

ou consulter [la version en ligne ici](#)

Pour exécuter les notebooks de démonstration, ouvrez Jupyter Notebook et parcourez le répertoire notebooks/.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
