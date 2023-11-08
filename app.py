# LIBRAIRIES

import os
import numpy as np
import streamlit as st


from src.streamlit.mods.utils import *
from src.streamlit.mods.styles import *
from src.streamlit.mods.explain import *
import src.models as lm

st.set_page_config(**page_config)
load_css('src/streamlit/mods/styles.css')



lm.models.RECORD_DIR='models/records'
lm.models.FIGURE_DIR='reports/figures'

trainer_modeles = load_all_models()

# Graphs 


choose = setup_sidebar()
# PAGES

import src.streamlit as ls

if choose == "Introduction":
    ls.a_intro.content(st)

if choose == "Jeux de données":
    ls.b_dataset.content(st)

if choose == "Modélisation":
   ls.c_modelisation.content(st)

if choose == "Interprétabilité et Segmentation" :
    ls.d_segmentation.content(st)

if choose == "Utilisation du modèle":
    ls.e_demo.content(st,trainer_modeles)

if choose == "Conclusion":
    ls.f_conclusion.content(st,trainer_modeles)