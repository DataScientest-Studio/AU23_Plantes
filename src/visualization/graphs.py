from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import src.features as lf
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns

import tensorflow.keras as keras
from keras.models import Model

def history_graph(history1 : dict, history2:dict =None) -> None :
    """
    Generate a history graph using the provided training history.
    Parameters:
    - `history1` (dict): The training history for the first round.
    - `history2` (dict, optional): The training history for the second round. Default is None.
    Returns:
    - None
    This function plots the training and validation accuracy as well as the training and validation loss using the provided training history. It uses the `plot_accuracy` and `plot_loss` helper functions to plot the graphs. If a second training history is provided, it plots the graphs for both rounds.
    Example usage:
    ```python
    history_graph(history1, history2)
    ```
    """
    min = 0.6
    def plot_accuracy(history:dict, round_str:str):
        plt.plot(history['main_categorical_accuracy'], label='Training Accuracy')
        plt.plot(history['val_main_categorical_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min, 1])
        plt.title(round_str + 'Training and Validation Accuracy')

    def plot_loss(history:dict, round_str:str):
        plt.plot(history['main_loss'], label='Training Loss')
        plt.plot(history['val_main_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.xlabel('epoch')
        plt.title(round_str + 'Training and Validation Loss')

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plot_accuracy(history1, 'First round')
    plt.subplot(2, 2, 2)
    plot_loss(history1, 'First round')
    if (history2) :
        plt.subplot(2, 2, 3)
        plot_accuracy(history2, 'Second round')
        plt.subplot(2, 2, 4)
        plot_loss(history2, 'Second round')
    plt.show()


def plot_confusion_matrix(results:str,description:str="" ):
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
    plt.title(f"Confusion Matrix\n{description}")
    plt.show()



def display_results(results:pd.DataFrame,nb:int=15,gradcam:bool=False, model:Model=None, img_size:tuple=None):
    """
    Display the results of a classification model.
    Parameters:
    - results (pandas DataFrame): The results dataframe containing the predicted and actual labels.
        (from data_builder.get_predictions_dataframe())
    - nb (int): The number of results to display. Default is 15.
    - gradcam (bool): Whether to show GradCAM layer or not. Default is False.
    - model (object): The classification model. Required if `gradcam` is True.
    - img_size (tuple): The size of the image. Required if `gradcam` is True.
    Returns:
    - None
    """
    results_df = results.reset_index(drop=True)
    plt.figure(figsize = (20 , 20))
    n = 0
    for i in range(nb):
        n+=1
        plt.subplot(math.ceil(16/5) , 5, n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.3)
        if gradcam :
            image = lf.data_builder.gradCAMImage(model, f"{results_df.filename[i]}", img_size)
        else :    image = lf.data_builder.readImage(f"{results_df.filename[i]}")
        plt.imshow(image)
        plt.title(f'{results_df.actual[i]} \n Pred: {results_df.predicted[i]}')
