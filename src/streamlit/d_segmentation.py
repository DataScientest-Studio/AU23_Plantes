from src.streamlit.mods.histogram_color import compute

def content(st) :
    st.header("Interprétabilité et Segmentation")
    # st.markdown("")
    st.subheader('Segmentation sémantique des images')
    compute(st=st)
    return st