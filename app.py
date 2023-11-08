# LIBRAIRIES

import os
import numpy as np
import streamlit as st

page_config =  {
    "page_title": "FloraFlow",
    "page_icon": "🌱"
}

st.set_page_config(**page_config)

from src.streamlit.mods.styles import setup_sidebar
from src.streamlit.mods.utils import load_all_models,load_css
load_css('src/streamlit/mods/styles.css')


import  src.models as lm

lm.models.RECORD_DIR='models/records'
lm.models.FIGURE_DIR='reports/figures'

trainer_modeles = load_all_models()

# Graphs 


choose = setup_sidebar()
# PAGES

import src.streamlit as ls

if choose == "Introduction":
    ls.a_intro.content()

if choose == "Jeux de données":
    ls.b_dataset.content()

if choose == "Modélisation":
    ls.c_modelisation.content()

if choose == "Interprétabilité et Segmentation" :
    ls.d_segmentation.content()

if choose == "Utilisation du modèle":
    ls.e_demo.content(trainer_modeles)

if choose == "Conclusion":
    ls.f_conclusion.content(trainer_modeles)