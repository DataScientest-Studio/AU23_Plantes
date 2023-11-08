

from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *

def content(st, trainer_modeles) :
    st.write("### Conclusion")
    nom_des_modeles = {
        'MobileNetv3': '4-dense-Mob',
        'ResNetv2': '4-dense-Res'
    }

    tab_titles = [f"Modèle {nom_des_modeles.get(type(model).__name__, 'Inconnu')}" for model in trainer_modeles]
    tabs = st.tabs(tab_titles)

    for model, tab in zip(trainer_modeles, tabs):
        with tab:
            model_class_name = type(model).__name__
            model_name_csv = nom_des_modeles.get(model_class_name, 'Inconnu')

            if model_name_csv == 'Inconnu':
                st.error(f"Le nom du modèle pour la classe {model_class_name} est inconnu.")
                continue

            conf_matrix = pred_confusion_matrix(feedbacks, especes, model_name_csv)
            fig = plot_confusion_matrix(conf_matrix, especes)
            st.pyplot(fig)

    return st