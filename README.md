# FloraFlow - Grow better with AI.

This GitHub repository contains all the code necessary for the proper functioning of the project we named FloraFlow. This cornerstone project was carried out during our Data Scientist training at Data Scientest.

## Project overview :
This project aims to develop a deep learning model capable of accurately classifying different plant species, with the goal of improving crop management.

The main steps of our project include:

- In-depth exploration of the dataset,
- Data preparation,
- The design and implementation of a learning model,
- And the analysis of its performance.

Our ultimate goal is to provide the foundations of a robust tool for plant classification, thereby offering farmers more efficient means to manage their crops.

Behind FloraFlow are four individuals in training at DataScientest:

- Gilles de Peretti [Linkedin](https://www.linkedin.com/in/gilles-de-peretti-8219425a/) - [GitHub](https://github.com/gillesdeperetti)
- Hassan ZBIB [Linkedin](https://www.linkedin.com/in/zbib-hassan-a34573272/) - [GitHub](https://github.com/Haszb)
- Dr. Iréné AMIEHE ESSOMBA [Linkedin](https://www.linkedin.com/in/amiehe-essomba "Amiehe Essomba") - [GitHub](https://github.com/amiehe-essomba "Amiehe Essomba")
- Olivier MOTELET [Linkedin](#) - [GitHub](#)

## About the dataset: 

We used the Kaggle's dataset [V2 Plant Seedlings Dataset](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset), which includes a collection of plant images belonging to various species.

The V2 Plant Seedlings dataset consists of 5539 images representing seedlings at the germination stage. They are grouped into 12 classes, each representing a variety/species of plants.

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

To start using FloraFlow, clone the repository and install the necessary dependencies:

    ```bash
    git clone https://github.com/DataScientest-Studio/AU23_Plantes.git FloraFlow
    cd FloraFlow
    pip install -r requirements.txt
    ```

Then, download the Kaggle dataset mentioned earlier and drop it into your FloraFlow folder. It's best to avoid using the NonSegmented folder, as it might have pics of several plant species all mixed in one folder.

## Utilisation

To get the Streamlit app going and interact with the user interface:

```bash
streamlit run streamlit/app.py 
```

or consult [the online version](#)

To run the demo notebooks, just open Jupyter Notebook and navigate through the notebooks/ directory.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
