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

images_especes = "src/streamlit/fichiers/images_especes.png"
color_palette = px.colors.sequential.speed
data = pd.read_csv("src/streamlit/fichiers/dataset_plantes.csv")
feedbacks = "src/streamlit/fichiers/feedback/feedback.csv"
especes = list(data['Classe'].unique())

def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def distribution_des_classes(data):
    return px.histogram(data, x="Classe", title="Distribution des classes", color='Classe', color_discrete_sequence=color_palette)

@st.cache_data
def poids_median_resolution(data):
    df_median = data.groupby('Classe')['Résolution'].median().reset_index()
    return px.bar(df_median, x='Classe', y='Résolution', title="Résolution médiane des images selon l'espèce", color='Classe', color_discrete_sequence=color_palette)

@st.cache_data
def ratios_images(data):
    bins = [0, 0.90, 0.95, 0.99, 1.01, 1.05, max(data['Ratio']) + 0.01]
    bin_labels = ['<0.90', '0.90-0.94', '0.95-0.99', '1', '1.01-1.04', '>1.05']
    data['Ratio_cat'] = pd.cut(data['Ratio'], bins=bins, labels=bin_labels, right=False)
    count_data = data.groupby(['Ratio_cat', 'Classe']).size().reset_index(name='Nombre')
    return px.bar(count_data, x='Ratio_cat', y='Nombre', color='Classe', title="Nombre d'images par classe pour chaque catégorie de ratio", barmode='group', color_discrete_sequence=color_palette)

@st.cache_data
def repartition_rgb_rgba(data):
    compte_rgba = data[data["Canaux"] == 4].shape[0]
    compte_rgb = data[data["Canaux"] == 3].shape[0]
    valeurs = [compte_rgb, compte_rgba]
    etiquettes = ["RGB", "RGBA"]
    return px.pie(values=valeurs, names=etiquettes, title="Répartition des images en RGB et RGBA", color=etiquettes, color_discrete_sequence=color_palette)

@st.cache_data
def repartition_especes_images_rgba(data):
    donnees_rgba = data[data["Canaux"] == 4]
    repartition_especes_rgba = donnees_rgba['Classe'].value_counts().reset_index()
    repartition_especes_rgba.columns = ['Classe', 'Nombre']
    return px.bar(repartition_especes_rgba, x='Classe', y='Nombre', title='Répartition des espèces au sein des images RGBA', color='Classe', color_discrete_sequence=color_palette)

@st.cache_resource()
def load_all_models(model_names):
    models = {}
    for model_name in model_names:
        model_path = os.path.join('models/records/final-test2', model_name)
        models[model_name] = load_model(model_path)
    return models

def preprocess_image(image, target_size=(224, 224), color='white', kernel=2.0, threshold=[0, 80]):
    image = image.resize(target_size)
    image = np.array(image.convert("RGB")).astype('float32')
    
    if image.dtype == 'float32':
        norm=True
        img_without_bg = (image - np.min(image)) / (np.max(image) - np.min(image))

    img_without_bg = lf.segmentation.remove_background(
                        x=tf.constant(img_without_bg, dtype=tf.float32), 
                        color=color, radius=kernel,
                        threshold=threshold
                     )

    if norm:
        image = img_without_bg * (np.max(image) - np.min(image)) + np.min(image)

    if isinstance(image, tf.Tensor):
        image = image.numpy()
        if image.ndim == 4 and img_without_bg.shape[0] == 1:
            image = np.squeeze(img_without_bg, axis=0)

    if image.ndim == 2:
        image = np.expand_dims(img_without_bg, axis=-1)

    image = np.expand_dims(img_without_bg, axis=0)
    st.image(image)
    return image

#def preprocess_image_safe(image, target_size=(150, 150)):
#        image = image.resize(target_size)
 #       image = image.convert("RGB")
  #      image_np = img_prep.img_to_array(image) / 255.0  
   #     image_np = np.expand_dims(image_np, axis=0)
    #    return image_np


def enregistrer_feedback_pandas(url, classe_predite, bonne_classe, nom_modele):
    file_path = os.path.join('src', 'streamlit', 'fichiers', 'feedback', 'feedback.csv')
    df = pd.DataFrame([[url, classe_predite, bonne_classe, nom_modele]], columns=['URL', 'Classe Predite', 'Bonne Classe', 'Modèle Utilisé'])
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
    st.session_state['resultat'] = None
    st.session_state['id_classe_predite'] = None
    st.session_state['source_image'] = None


def pred_confusion_matrix(feedback, classes, model_name=None):
    df_feedback = pd.read_csv(feedback)
    if model_name is not None:
        df_feedback = df_feedback[df_feedback['Modèle Utilisé'] == model_name]
    df_feedback.dropna(subset=['Classe Predite', 'Bonne Classe'], inplace=True)
    df_feedback.drop_duplicates(subset=['URL', 'Classe Predite', 'Bonne Classe', 'Modèle Utilisé'], inplace=True)
    matrice_conf = confusion_matrix(df_feedback['Bonne Classe'], df_feedback['Classe Predite'], labels=classes)
    return matrice_conf

def plot_confusion_matrix(conf_matrix, classes):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes, cmap='Blues', ax=ax)
    ax.set_title("Matrice de confusion prédictions sur les images du web")
    ax.set_ylabel('Vraie classe')
    ax.set_xlabel('Classe prédite')
    return fig