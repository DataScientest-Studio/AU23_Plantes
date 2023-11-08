import streamlit as st
from src.streamlit.mods.histogram_color import compute


def content() :
    st.header("Interprétabilité et Segmentation")
    # st.markdown("")
    st.subheader('Segmentation sémantique des images')
    compute(st=st)
