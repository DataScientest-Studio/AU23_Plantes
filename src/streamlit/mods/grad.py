import streamlit as st 
import src.features as lf
import tensorflow as tf 
import numpy as np
import matplotlib.cm as cm
import cv2
import matplotlib.pyplot as plt 

from skimage.transform import resize
from src.features.data_builder import deprocess_image, guided_backprop, make_gradcam_heatmap


def gradCAMImage(img :np.ndarray, img_size : tuple=(224, 224), model=None, base_model_wrapper=None,
                segmented : bool=False, guidedGrad_cam:bool=False ) -> np.ndarray:
    """
    Generates a Grad-CAM image by overlaying the heatmap on the original image.

    Args:
        img_path: The path to the input image.
        img_size: The desired size of the input image.
        model: The neural network model used for generating the heatmap.
        base_model_wrapper: The associated base model wrapper.

    Returns:
        The superimposed image with the heatmap.
    """
    expended_img = img

    #expended_img      = np.uint8(255 * expended_img)

    heatmap = make_gradcam_heatmap(expended_img, model, base_model_wrapper)
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    if segmented:
        img = resize(img, output_shape=img_size)
        img = lf.segmentation.remove_background(x=img)
        img = tf.keras.utils.img_to_array(img)
        img = np.uint8(img * 255)
        IMG = img.copy( )
    else:
        img = np.uint8(img * 255)
        IMG = img.copy( )

    shape           = (1,) + img.shape
    IMG             = IMG.reshape(shape)
    gb              = guided_backprop(model=model, img=IMG, upsample_size=img_size) 
    guided_gradcam  = deprocess_image(gb * jet_heatmap)
    guided_gradcam  = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)

    if guidedGrad_cam:
        return guided_gradcam
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

def grad_cam(st:st) -> None:
    pass