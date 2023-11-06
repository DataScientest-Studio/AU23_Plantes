def button_style(st):
    st.write(
        f'<style>div.stButton > button{{background-color: white; color: black; \
            padding: 10px 20px, text-align: center;\
            display: inline-block;\
            font-size: 16px;\
            border-radius: 50px;\
            background-image: linear-gradient(to bottom, red, darkorange, orange);\
            font-weight: bolder}} </style>',
        unsafe_allow_html=True
    )

def text_style():
    custom_css_title = """
    .text {
        color: black; /* Couleur du texte */
        background-color: white; /* Couleur de l'arrière-plan */
        font-size: 15px; /* Taille de police */
        text-decoration: None; /* Souligné underline overline */
        margin: 10px; /* Marge extérieure */
        border-radius: 5px; /* Coins arrondis */
        font-family: Arial, sans-serif; /* font family*/
        text-align: justify;
        }
    """

def text_transform(st, text):
    st.write('<style>{}</style>'.format(text_style()), unsafe_allow_html=True)
    st.markdown(f'<p class="text">{text}</p>', unsafe_allow_html=True)