
import streamlit as st
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm
import src.features as lf
st.write(especes)
def content( trainer_modeles) :
    nom_des_modeles_csv = {
        'MobileNetv3': '4-dense-Mob',
        'ResNetv2': '4-dense-Res'
    }

    nom_des_modeles_affichage = {
        'MobileNetv3': 'MobileNetv3Large',
        'ResNetv2': 'ResNetv2'
    }

    tab_titles = [nom_des_modeles_affichage.get(type(model).__name__, 'Modèle Inconnu') for model in trainer_modeles]
    tabs = st.tabs(tab_titles)

    for model, tab in zip(trainer_modeles, tabs):
        with tab:
            model_class_name = type(model).__name__
            model_name_csv = nom_des_modeles_csv.get(model_class_name, 'Inconnu')

            if model_name_csv == 'Inconnu':
                st.error(f"Le nom CSV du modèle pour la classe {model_class_name} est inconnu.")
                continue

            conf_matrix = pred_confusion_matrix(feedbacks, especes, model_name_csv)
            fig = plot_confusion_matrix(conf_matrix, especes)
            st.pyplot(fig)

