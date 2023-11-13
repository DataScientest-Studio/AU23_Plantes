import streamlit as st
import time
import requests
from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm
import src.features as lf
from streamlit_cropper import st_cropper


data = pd.read_csv("src/streamlit/fichiers/dataset_plantes.csv")
especes = list(data['Classe'].unique())


def content(trainer_modeles) :
    st.header('Utilisation du modèle')
    url = None
    selected_image = None
    trainer_modeles = load_all_models()
    aspect_ratio = (1, 1)
    box_color = '#e69138'
    modeles = {'Modèle MobileNetV3Large': 0, 'Modèle ResNet50V2': 1}
    choix_nom_modele = st.selectbox("Choisissez un modèle", list(modeles.keys()))
    choix_idx = modeles[choix_nom_modele]
    choix_modele = trainer_modeles[choix_idx]
    especes = list(data['Classe'].unique())
    if 'show_elements' not in st.session_state:
        st.session_state['show_elements'] = False  
    option = st.selectbox("Comment voulez-vous télécharger une image ?",
                          ("Lot d'images de test", "Depuis mon ordinateur", "A partir d'une URL"))
    image = None
    progress_bar = st.progress(0)
    if option == 'Depuis mon ordinateur':
        with st.expander("🎨 Guide visuel pour le choix d'images parfaites"):
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
    elif option == "A partir d'une URL":
        with st.expander("🎨 Guide visuel pour le choix d'images parfaites"):
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
    elif option == "Lot d'images de test":
        with st.expander("ℹ️ Informations"):
            show_information_block("gallery_info")
        gallery_path = 'src/streamlit/fichiers/gallery'
        gallery_list = os.listdir(gallery_path)
        image_extensions = ['.jpeg', '.jpg', '.png']
        image_list = [file for file in gallery_list if os.path.splitext(file)[1].lower() in image_extensions]
        selected_image = st.selectbox("Sélectionnez une image", image_list)
        image_path = os.path.join('src/streamlit/fichiers/gallery', selected_image)
        image = Image.open(image_path)
        st.session_state['loaded'] = True
        st.write("##### Aperçu de l'image à prédire")
        tab1, tab2, tab3 = st.columns([0.25,0.5,0.25])
        with tab1:
            st.write("")
        with tab2:
            st.image(image, width=350)
        with tab3:
            st.write("")

    feedback_placeholder = st.empty()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🚥 Prédire l’espèce 🚥", use_container_width=True) and image is not None and st.session_state['loaded']:
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
                st.markdown(f"Ce résultat est-il correct ?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Oui, c’est la bonne espèce"):
                        image_info = url if st.session_state.get('source_image') == 'url' else selected_image
                        enregistrer_feedback_pandas(image_info, st.session_state['classe_predite'],
                                                    st.session_state['classe_predite'], trainer_modeles[choix_idx])
                        st.session_state['feedback_soumis'] = True
                        st.session_state['classe_predite'] = None
                with col2:
                    if st.button("👎 Non…"):
                        st.session_state['mauvaise_pred'] = True
                if st.session_state.get('mauvaise_pred'):
                    bonne_classe = st.multiselect("Précisez la bonne classe :", especes)
                    if st.button('Confirmer la classe'):
                        if bonne_classe:
                            image_info = url if st.session_state.get('source_image') == 'url' else selected_image
                            enregistrer_feedback_pandas(image_info, st.session_state['classe_predite'], bonne_classe[0],
                                                        trainer_modeles[choix_idx])
                            time.sleep(3)
                            feedback_placeholder.empty()
                            reset_state()
                        else:
                            st.error("Veuillez sélectionner une classe avant de confirmer.")
    st.divider()
    st.header('Performance du modèle en contexte d’utilisation')
    if st.button('Afficher les matrices de confusion'):
        st.session_state['show_elements'] = not st.session_state['show_elements']
    if st.session_state['show_elements']:
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