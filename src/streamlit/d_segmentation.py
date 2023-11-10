from src.streamlit.mods.histogram_color import compute

def content() :
   import streamlit as st
   import matplotlib.pyplot as plt 

   st.title("Segmentation & Interprétabilité")
   st.header('Segmentation sémantique des images')
   compute(st=st)
   st.header('Interprétabilité')

   path1 = "./reports/figures/test-classifpatient/classif-256-0.4_sample_graphs_segmented_gradcam_guidedGradcam.png"
   path2 = "./reports/figures/test-classifpatient/classif-256-0.4_sample_graphs_gradcam_guidedGradcam.png"

   with st.expander("Images non-segmentatées"):
      im1 = plt.imread(path2)
      st.image(im1)
   with st.expander("Images Segmentées"):
       im2 = plt.imread(path1)
       st.image(im2)