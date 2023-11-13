import streamlit as st 
import plotly.express as px
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image as img_prep
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import os
import src.features as lf 
import tensorflow as tf
import src.models as lm
import streamlit.components.v1 as html
import re


images_especes = "src/streamlit/fichiers/images_especes.png"
color_palette = px.colors.sequential.speed
color_palette_b = ['#E9D98B', '#66940A']
feedbacks = "src/streamlit/fichiers/feedback/feedback.csv"

@st.cache_data()
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data()
def distribution_des_classes(data):
    return px.histogram(data, x="Classe", title="Distribution des classes", color='Classe', color_discrete_sequence=color_palette)

@st.cache_data()
def poids_median_resolution(data):
    df_median = data.groupby('Classe')['Poids'].median().reset_index()
    return px.bar(df_median, x='Classe', y='Poids', title="Poids médian des images par classe", labels={'Poids': 'Poids Médian'}, color='Classe', color_discrete_sequence=color_palette)

@st.cache_data()
def ratios_images(data):
    data['Ratio'] = data['Largeur'] / data['Hauteur']
    bins = [0, 0.90, 0.95, 0.99, 1.01, 1.05, max(data['Ratio']) + 0.01]
    bin_labels = ['<0.90', '0.90-0.94', '0.95-0.99', '1', '1.01-1.04', '>1.05']
    data['Ratio_cat'] = pd.cut(data['Ratio'], bins=bins, labels=bin_labels, right=False)
    count_data = data.groupby(['Ratio_cat', 'Classe']).size().reset_index(name='Nombre')
    return data, px.bar(count_data, x='Ratio_cat', y='Nombre', color='Classe', title="Nombre d'images par classe pour chaque catégorie de ratio", barmode='group', color_discrete_sequence=color_palette)

@st.cache_data()
def repartition_rgb_rgba(data):
    data['Canaux'] = data['Forme'].str.extract(r'\((?:\d+, ){2}(\d+)\)')[0].astype(int)
    compte_rgb = (data['Canaux'] == 3).sum()
    compte_rgba = (data['Canaux'] == 4).sum()
    valeurs = [compte_rgb, compte_rgba]
    etiquettes = ["RGB", "RGBA"]
    return data, px.pie(values=valeurs, names=etiquettes, title="Répartition des images en RGB et RGBA", color_discrete_sequence=color_palette_b)

@st.cache_data()
def repartition_especes_images_rgba(data):
    data['Canaux'] = data['Forme'].str.extract(r'\((?:\d+, ){2}(\d+)\)')[0].astype(int)
    donnees_rgba = data[data["Canaux"] == 4]
    repartition_especes_rgba = donnees_rgba['Classe'].value_counts().reset_index()
    repartition_especes_rgba.columns = ['Classe', 'Nombre']
    return data, px.bar(repartition_especes_rgba, x='Classe', y='Nombre', title='Répartition des espèces au sein des images RGBA', color='Classe', color_discrete_sequence=color_palette_b)

@st.cache_resource()
def load_all_models():
    trainer_models = [lm.final_test.stage4.MobileNetv3(lf.data_builder.VoidSeedlingImageDataWrapper(), 'final-test2'),
                      lm.final_test.stage4.ResNetv2(lf.data_builder.VoidSeedlingImageDataWrapper(), 'final-test2')]
    for t in trainer_models:
        t.fit_or_load(training=False)
    return trainer_models


def enregistrer_feedback_pandas(image_info, classe_predite, bonne_classe, nom_modele):
    file_path = os.path.join('src', 'streamlit', 'fichiers', 'feedback', 'feedback.csv')
    
    url = image_info if isinstance(image_info, str) and image_info.startswith('http') else None
    df = pd.DataFrame([[url, classe_predite, bonne_classe, nom_modele]], 
                      columns=['URL', 'Classe Predite', 'Bonne Classe', 'Modèle Utilisé'])
    
    if os.path.isfile(file_path):
        df_existante = pd.read_csv(file_path)
        df_finale = pd.concat([df_existante, df], ignore_index=True)
    else:
        df_finale = df
    
    df_finale.to_csv(file_path, index=False)
    st.success('Feedback enregistré avec succès !')


def reset_state():
    st.session_state['feedback_soumis'] = False
    st.session_state['mauvaise_pred'] = False
    st.session_state['classe_predite'] = None
    st.session_state['source_image'] = None


def pred_confusion_matrix(feedback, classes, model_name=None):
    df_feedback = pd.read_csv(feedback)
    
    pattern = re.compile(r'shepherd[’\']?s?\s*purse', re.I)

    df_feedback['Bonne Classe'] = df_feedback['Bonne Classe'].apply(lambda x: re.sub(pattern, 'Shepherd\'s Purse', x))
    df_feedback['Classe Predite'] = df_feedback['Classe Predite'].apply(lambda x: re.sub(pattern, 'Shepherd\'s Purse', x))
    
    if model_name is not None:
        df_feedback = df_feedback[df_feedback['Modèle Utilisé'] == model_name]
    
    if 'Shepherd\'s Purse' not in classes:
        classes.append('Shepherd\'s Purse')

    matrice_conf = confusion_matrix(df_feedback['Bonne Classe'], df_feedback['Classe Predite'], labels=classes)
    return matrice_conf

@st.cache_data()
def plot_confusion_matrix(conf_matrix, classes):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes, cmap='Greens', ax=ax)
    ax.set_ylabel('Vraie classe')
    ax.set_xlabel('Classe prédite')
    return fig

@st.cache_data()
def get_plotly_html(file) :
    with open(file, "r", encoding='utf-8') as f:
        html_content = f.read()
    return html_content