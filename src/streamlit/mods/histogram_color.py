import src.features as lf 
import plotly.exceptions as px 
import pandas as pd 
import sys, os
import numpy as np 
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
import streamlit as st
import tensorflow as tf

def get_cmap_list(index : int = 10):
    c = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 
        'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 
        'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 
        'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 
        'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 
        'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 
        'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 
        'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 
        'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 
        'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 
        'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 
        'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
        'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 
        'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r'
        , 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 
        'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
        'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
        'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 
        'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
        'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 
        'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
        'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
        'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 
        'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
        'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 
        'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 
        'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
        ]
    
    return c[index]

def plant_names():
    names, PATHS    = [], []
    PATH            = "./src/streamlit/fichiers/gallery"

    for x in [f for f in os.listdir(PATH) if f.endswith(('.jpg', '.png', '.jpeg'))]:
        if "black" not in  x :
            if  "loose" not in x:
                n = x.split("_")
                name = ""
                for i, _name_ in enumerate(n[:-1]):
                    _name_ :str = _name_.rstrip().lstrip().capitalize()

                    if i < len(n[:-1])-1:  name += _name_ + "-"
                    else: name += _name_
 
                names.append(name)
                PATHS.append(os.path.join(PATH, x))
    
    return names, PATHS

def plotly_histogram(
    channels: np.ndarray, 
    size    : int = 0.05, 
    opacity : float=0.9,
    name    :  str = ''
    )-> px:
    
    import plotly.express as px
    from plotly.graph_objs import Image


    # Données pour les histogrammes
    x1 = channels[0]
    x2 = channels[1]
    x3 = channels[2]

    colormap_r = px.colors.sequential.gray[:1] + px.colors.sequential.gray_r + px.colors.sequential.gray + px.colors.sequential.gray_r 
    colormap_g = px.colors.sequential.Greens_r + px.colors.sequential.Purpor + px.colors.sequential.Magenta_r * 2
    colormap_b = px.colors.sequential.Blues_r +  px.colors.sequential.YlOrRd_r + px.colors.sequential.YlOrRd_r[4:] + px.colors.sequential.YlOrRd_r[7:]

    norm       = ['', 'percent', 'probability', 'density', 'probability', "density"]
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=(
                            'Luminance', 
                            'Vert au Magenta', 
                            'Bleue au Jaune')
                        )

    # Ajoutez les histogrammes sur chaque axe
    fig.add_trace(
                go.Histogram(
                            x=x1, name='<span style="color:black">Canal L</span>', 
                            xcalendar='persian', hoverlabel=dict(bgcolor="rgb(0,0,0)"),
                            marker=dict(color=colormap_r), opacity=opacity, autobinx=False,
                            xbins=dict(start=0, end=1, size=size)
                            ), row=1, col=1
                )
    fig.add_trace(
                go.Histogram(x=x2, name='<span style="color:darkgreen">Canal A</span>', 
                            xcalendar="ethiopian", hoverlabel=dict(bgcolor="rgb(0,255,0)"),
                            marker=dict(color=colormap_g), opacity=opacity, autobinx=False,
                            xbins=dict(start=0, end=1, size=size)
                            ), row=1, col=2
                )
    fig.add_trace(
                go.Histogram(x=x3, name='<span style="color:darkblue">Canal B</span>',
                            xcalendar="ethiopian", hoverlabel=dict(bgcolor="rgb(0,0,255)"),
                            marker=dict(color=colormap_b),opacity=opacity, autobinx=False,
                            xbins=dict(start=0, end=1, size=size)
                            ), row=1, col=3)


    # Personnalisez le layout
    fig.update_layout(
        title_text=f"{name} : Histogrammes de couleurs à 3 canaux dans l'espace colorimétrique LAB",
        xaxis=dict(title='Luminosité'),
        yaxis=dict(title="Intensité (Px)"),
        xaxis2=dict(title='Luminosité'),
        xaxis3=dict(title='Luminosité'),
    )


    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #fig.write_html('plotly.html')

    return fig

def compute(
        st              : st=None, 
        name_id         : int = 0,
        scaling_factor  : float = 1.0,
        size            : float = 0.04,
        opacity         : float = 0.9,
        show            : bool = False
        ):

    
    PLANT_NAMES, PATHS = plant_names()

    if show is True : 
        img = plt.imread(PATHS[name_id]).astype("float32")
        name = PLANT_NAMES[name_id]
        img2rgbLAB = lf.segmentation.RGB2LAB_SPACE(image=img.copy())
        channels = [img2rgbLAB[..., j].ravel() * scaling_factor for j in range(3)] 

        fig = plotly_histogram(channels=channels, size=size, opacity=opacity, name=name)
        fig.show()
        
        img_without_bg, mask = lf.segmentation.\
                    remove_background(
                        x=tf.constant(img, dtype=tf.float32), radius=4, mask=True )
        
        cmap = get_cmap_list()
        fig2, axes= plt.subplots(1,4, figsize=(12, 3))
        titles = ["Image originale, dans l'espace RGB", "Image dans L'espace LAB", 'Masque', 'Image Segmentée']
        for i in range(4):
            axes[i].axis('off')
            axes[i].set_title(titles[i], fontsize='medium')

        axes[0].imshow(img, interpolation='nearest')
        axes[1].imshow(img2rgbLAB[..., 1], interpolation='nearest', cmap=cmap)
        axes[2].imshow(mask, interpolation='nearest')
        axes[3].imshow(img_without_bg, interpolation='nearest')

        fig2.suptitle(f"{name} : Processus de suppression d'arrière plan", fontsize="large", weight="bold")
        plt.show()

    else:
        plant_id = st.select_slider("ID de la Plante", options=range(len(PATHS)), value=0)
        st.write("Nom de la plante : ", PLANT_NAMES[plant_id])

        threshold = st.select_slider('Domaine de pixellisation', options=[round(x, 2) for x in np.arange(0, 1.005, 0.01)], value=(0, 0.33))
        st.write('Valeurs:', threshold)
        threshold = [int(p * 255) for p in threshold]

        col2, col3, col4 = st.columns(3)

        with col2:
            color = st.selectbox("Couleur de fond", options=('white', "black"))
            st.write('Valeur:',  color)

        with col3:
            kernel = st.slider("Kernel", min_value=1, value=1, step=1, max_value=10)
            st.write('Valeurs:', (kernel, kernel))

        with col4:
            show_seg = st.checkbox('Plus de détails')
            st.write('état:', show_seg)

        img = plt.imread(PATHS[plant_id]).astype("float32")
        name = PLANT_NAMES[plant_id]

        img2rgbLAB = lf.segmentation.RGB2LAB_SPACE(image=img.copy())
        channels = [img2rgbLAB[..., j].ravel() * scaling_factor for j in range(3)] 

        img_without_bg, mask = lf.segmentation.\
                    remove_background(
                        x=tf.constant(img, dtype=tf.float32), 
                        color=color, radius=kernel,
                        threshold=threshold, mask=True
                        )

        if st.button('run'):
            fig1 = plotly_histogram(channels=channels, size=size, opacity=opacity, name=name)
            st.plotly_chart(fig1, use_container_width=True)

            if show_seg:
                cmap = get_cmap_list()
                fig2, axes= plt.subplots(1,4, figsize=(12, 3))
                titles = ["Image dans l'espace RGB", "Image dans l'espace LAB", 'Masque', 'Image Segmentée']
                for i in range(4):
                    axes[i].axis('off')
                    axes[i].set_title(titles[i], fontsize='medium')

                axes[0].imshow(img, interpolation='nearest')
                axes[1].imshow(img2rgbLAB[..., 1], interpolation='nearest', cmap=cmap)
                axes[2].imshow(mask, interpolation='nearest')
                axes[3].imshow(img_without_bg, interpolation='nearest')

                fig2.suptitle(f"{name} : Background Removal Process", fontsize="large", weight="bold")
                st.pyplot(fig2)

            



