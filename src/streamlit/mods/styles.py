from streamlit_option_menu import option_menu
import streamlit as st

linkedin_icon = "src/streamlit/fichiers/linkedin_icon.png"
github_icon = "src/streamlit/fichiers/github_icon.png"
ds_logo = "src/streamlit/fichiers/logo-2021.png"
logo = "src/streamlit/fichiers/FloraFlow.png"

page_config = {
    "page_title": "FloraFlow",
    "page_icon": "🌱"
}

def setup_sidebar():
    with st.sidebar:
        col1, col2, col3 = st.columns([0.2,0.7,0.2])
        with col2 : 
            st.image(logo)  
        choose = option_menu("", ["Introduction", "Jeu de données",  "Modélisation", "Interprétabilité & Segmentation", "Utilisation du modèle", "Conclusion"],
                            icons=['arrow-repeat', 'database',  'box', 'layers-half', 'cloud-download', 'check-circle'],
                            default_index=0,
                            styles={
            "container": {"padding": "1!important", "background-color": "#fafafa"},
            "icon": {"color": "#FCA500", "font-size": "24px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#68AD3A"},
        }
        )
        footer = '''
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

        # Equipe du projet
        '''

        st.sidebar.markdown(footer, unsafe_allow_html=True)


        st.sidebar.markdown("""
        #### Gilles de Peretti <a href="https://github.com/gillesdeperetti" target="_blank"><i class='fab fa-github'></i></a> <a href="https://www.linkedin.com/in/gilles-de-peretti-8219425a/" target="_blank"><i class='fab fa-linkedin'></i></a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("""
        #### Hassan ZBIB <a href="URL_GITHUB_HASSAN" target="_blank"><i class='fab fa-github'></i></a>  <a href="URL_LINKEDIN_HASSAN" target="_blank"><i class='fab fa-linkedin'></i></a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("""
        #### Dr. Iréné AMIEHE ESSOMBA <a href="URL_GITHUB_IRENE" target="_blank"><i class='fab fa-github'></i></a> <a href="URL_LINKEDIN_IRENE" target="_blank"><i class='fab fa-linkedin'></i></a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("""
        #### Olivier MOTELET <a href="https://github.com/nagoli" target="_blank"><i class='fab fa-github'></i></a>  <a href="https://www.linkedin.com/in/oli-mo-ux/" target="_blank"><i class='fab fa-linkedin'></i></a>
        """, unsafe_allow_html=True)
        st.write("""


        """)
        st.sidebar.image(ds_logo)  
        return choose 
