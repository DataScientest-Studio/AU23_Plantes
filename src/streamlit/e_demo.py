import streamlit as st
import time
import requests
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm
from streamlit_cropper import st_cropper

def content(trainer_modeles) :
    url = None
    selected_image = None
    trainer_modeles = load_all_models()
    aspect_ratio = (1, 1)
    box_color = '#e69138'
    modeles = {'Modèle MobileNetV3Large': 0, 'Modèle ResNet50V2': 1}
    choix_nom_modele = st.selectbox("Choisissez un modèle", list(modeles.keys()))
    choix_idx = modeles[choix_nom_modele]
    choix_modele = trainer_modeles[choix_idx]

    option = st.selectbox("Comment voulez-vous télécharger une image ?",
                          ("Choisir une image de la galerie", "Télécharger une image", "Utiliser une URL"))
    image = None
    progress_bar = st.progress(0)
    if option == 'Télécharger une image':
        with st.expander("ℹ️ Informations"):
            show_information_block("basic_info")
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            base_width = 350
            w_percent = base_width / float(image.size[0])
            h_size = int(float(image.size[1]) * float(w_percent))
            resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Recadrez l'image chargée")
                cropped_img = st_cropper(resized_image, realtime_update=True, box_color=box_color,
                                         aspect_ratio=aspect_ratio)
            with col2:
                st.write("##### Aperçu de l'image à prédire")
                st.image(cropped_img, use_column_width=True)
    elif option == 'Utiliser une URL':
        with st.expander("ℹ️ Informations"):
            show_information_block("image_url")
        url = st.text_input("Entrez l'URL de l'image")
        if url:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            base_width = 350
            w_percent = base_width / float(image.size[0])
            h_size = int(float(image.size[1]) * float(w_percent))
            resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
            st.session_state['source_image'] = 'url'
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Recadrez l'image chargée")
                cropped_img = st_cropper(resized_image, realtime_update=True, box_color=box_color,
                                         aspect_ratio=aspect_ratio)
            with col2:
                st.write("##### Aperçu de l'image à prédire")
                st.image(cropped_img, use_column_width=True)
            image = cropped_img
    elif option == 'Choisir une image de la galerie':
        with st.expander("ℹ️ Informations"):
            show_information_block("gallery_info")
        gallery_list = os.listdir('src/streamlit/fichiers/gallery')
        selected_image = st.selectbox("Sélectionnez une image", gallery_list)
        image_path = os.path.join('src/streamlit/fichiers/gallery', selected_image)
        image = Image.open(image_path)
        st.write("##### Aperçu de l'image à prédire")
        st.image(image, use_column_width=True)
    feedback_placeholder = st.empty()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Prédiction", use_container_width=True) and image is not None:
            progress_bar = st.progress(0)
            progress_bar.progress(0.3)
            image = image.resize((224, 224))
            st.session_state['classe_predite'] = choix_modele.predict_image(image)
            progress_bar.progress(1.0)
            time.sleep(3)
            progress_bar.progress(0)
    with feedback_placeholder.container():
            if st.session_state['classe_predite'] is not None:
                st.markdown(f"Selon le modèle, il s'agit de l'espèce **{st.session_state['classe_predite']}**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Correcte"):
                        # Handle feedback for both URL and non-URL sources
                        image_info = url if st.session_state.get('source_image') == 'url' else selected_image
                        enregistrer_feedback_pandas(image_info, st.session_state['classe_predite'],
                                                    st.session_state['classe_predite'], trainer_modeles[choix_idx])
                        st.session_state['feedback_soumis'] = True
                        st.session_state['classe_predite'] = None
                with col2:
                    if st.button("Incorrecte"):
                        st.session_state['mauvaise_pred'] = True
                if st.session_state.get('mauvaise_pred'):
                    especes = list(data['Classe'].unique())
                    bonne_classe = st.multiselect("Précisez la bonne classe :", especes)
                    if st.button('Confirmer la classe'):
                        if bonne_classe:
                            # Handle feedback for both URL and non-URL sources
                            image_info = url if st.session_state.get('source_image') == 'url' else selected_image
                            enregistrer_feedback_pandas(image_info, st.session_state['classe_predite'], bonne_classe[0],
                                                        trainer_modeles[choix_idx])
                            time.sleep(3)
                            feedback_placeholder.empty()
                            reset_state()
                        else:
                            st.error("Veuillez sélectionner une classe avant de confirmer.")