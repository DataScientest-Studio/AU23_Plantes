from types import SimpleNamespace

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import src.features as lf
import src.models as lm

import math
from sklearn.metrics import confusion_matrix
import seaborn as sns

import tensorflow.keras as keras
from keras.models import Model


def plot_history_graph(history1: dict, history2: dict = None, record_name: str = None, show=True) -> plt:
    """
    Generate a history graph using the provided training history.
    Parameters:
    - `history1` (dict): The training history for the first round.
    - `history2` (dict, optional): The training history for the second round. Default is None.
    - `record_name` (str, optional): The name of the record. Default is None.
    Returns:
    - None
    This function plots the training and validation accuracy as well as the training and validation loss using the provided training history. It uses the `plot_accuracy` and `plot_loss` helper functions to plot the graphs. If a second training history is provided, it plots the graphs for both rounds.
    Example usage:
    ```python
    history_graph(history1, history2)
    ```
    """
    min = 0.6

    def plot_accuracy(history: dict, round_str: str):
        title = 'Accuracy (Training and Validation)'
        plt.plot(history['categorical_accuracy'], label='Training Accuracy')
        plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min, 1])
        if round_str:
            plt.title(f"{round_str}\n{title}")
        else:
            plt.title(f"{title}")

    def plot_loss(history: dict, round_str: str):
        title = 'Loss (Training and Validation)'
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.xlabel('epoch')
        if round_str:
            plt.title(f"{round_str}\n{title}")
        else:
            plt.title(f"{title}")

    if history2:
        round_str = 'First round'
    else:
        round_str = None
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plot_accuracy(history1, 'First round')
    plt.subplot(2, 2, 2)
    plot_loss(history1, 'First round')
    if history2:
        plt.subplot(2, 2, 3)
        plot_accuracy(history2, 'Second round')
        plt.subplot(2, 2, 4)
        plot_loss(history2, 'Second round')
    fig.suptitle(f"{record_name} – Loss and Accuracy")
    if (show) : plt.show()
    return plt


def plot_confusion_matrix(results: str, record_name: str = "", show=True) -> plt:
    """
    Generate a confusion matrix plot based on the provided results.

    Parameters:
    - results: The results object containing the actual and predicted values.
    (from data_builder.get_predictions_dataframe())

    Return:
    - None
    """
    cf = confusion_matrix(results.actual, results.predicted)
    plt.figure(figsize=(10, 8))
    labels = sorted(results.actual.unique())
    sns.heatmap(cf, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', vmax=10)
    plt.title(f"{record_name} – Confusion Matrix")
    if (show) : plt.show()
    return plt


def display_results(results: pd.DataFrame, nb: int = 15, gradcam: bool = False, model: Model = None, base_model_wrapper : lm.model_wrapper.BaseModelWrapper = None,
                    img_size: tuple = None, record_name: str = None, segmented:bool=False, guidedGrad_cam : bool = False, show=True) -> plt:
    """
    Display the results of a classification model.
    Parameters:
    - results (pandas DataFrame): The results dataframe containing the predicted and actual labels.
        (from data_builder.get_predictions_dataframe())
    - nb (int): The number of results to display. Default is 15.
    - gradcam (bool): Whether to show GradCAM layer or not. Default is False.
    - model (object): The classification model. Required if `gradcam` is True.
    - base_model_wrapper (lm.model_wrapper.ModelWrapper): The associated base model wrapper. Required if `gradcam` is True.
    - img_size (tuple): The size of the image. Required if `gradcam` is True.
    - record_name (str): The name of the record. Default is None
    Returns:
    - None
    """

    results_df = results.reset_index(drop=True)

    #nb += 6
    if nb <= 6:
        nrows, ncols = 1, nb 
    else:
        nrows = nb / 6 
        if nrows > nb // 6:
            nrows = (nb // 6) + 1
        else:
            nrows = nb // 6
        ncols  = 6
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(int(ncols * 2 + 1.), int(nrows * 2 + 1.)))
    axes = axes.ravel()

    n = -1
    for i in range(ncols):
        for j in range(nrows):
            n += 1
            if n < nb:
                axes[n].axis('off')
                if gradcam:
                    if guidedGrad_cam:
                        if n % 2 == 0:
                            image = lf.data_builder.gradCAMImage(f"{results_df.filename[i]}", img_size, model, base_model_wrapper, segmented=segmented )
                        else:
                            image = lf.data_builder.gradCAMImage(f"{results_df.filename[i]}", img_size, model, base_model_wrapper, 
                                                                                segmented=segmented, guidedGrad_cam=guidedGrad_cam )
                    else:
                        image = lf.data_builder.gradCAMImage(f"{results_df.filename[i]}", img_size, model, base_model_wrapper, segmented=segmented )
                else:
                    image = lf.data_builder.readImage(f"{results_df.filename[i]}")
                
                axes[n].imshow(image)
                if guidedGrad_cam:
                    if n % 2 == 0:
                        axes[n].set_title(f'True: {results_df.actual[i]} \n Pred: {results_df.predicted[i]}', fontsize='small')
                else:
                    axes[n].set_title(f'True: {results_df.actual[i]} \n Pred: {results_df.predicted[i]}', fontsize='small')
            else: axes[n].remove()

    if record_name:
        if gradcam:
            fig.suptitle(f"{record_name} – Result Samples with GradCAM", fontsize="x-large")
        else:
            fig.suptitle(f"{record_name} – Result Samples", fontsize="x-large")

    if (show) : plt.show()
    return plt


    """
    results_df = results.reset_index(drop=True)
    fig = plt.figure(figsize=(20, 20))
    n = 0
    for i in range(nb):
        plt.axis('off')
        n += 1
        plt.subplot(math.ceil(16 / 3), 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        if gradcam:
            image = lf.data_builder.gradCAMImage(f"{results_df.filename[i]}", img_size, model, base_model_wrapper )
        else:
            image = lf.data_builder.readImage(f"{results_df.filename[i]}")
        plt.imshow(image)
        plt.title(f'{results_df.actual[i]} \n Pred: {results_df.predicted[i]}', fontsize='medium')
    if record_name:
        if gradcam:
            fig.suptitle(f"{record_name} – Result Samples with GradCAM", fontsize="x-large")
        else:
            fig.suptitle(f"{record_name} – Result Samples", fontsize="x-large")
    plt.show()
    """

