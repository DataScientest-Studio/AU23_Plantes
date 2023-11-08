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

import plotly.express as px
import plotly.io as pio

plotly_template = 'plotly_white'
#result_12_color_sequence = ['#3182bd', '#9ecae1', '#e6550d', '#fdae6b', '#31a354', '#a1d99b', '#756bb1', '#bcbddc', '#7b4173', '#ce6dbd', '#393b79', '#6b6ecf']
result_12_color_sequence = [
  "#4299e1",
  "#b6e3f5",
  "#ff7f3f",
  "#fed3a1",
  "#38c172",
  "#c6f6d5",
  "#9f7aea",
  "#dcd6ff",
  "#9c27b0",
  "#efb7df",
  "#5a67d8",
  "#8b9cf6"
]
continuous_cmap = sns.light_palette("#38c172", as_cmap=True)

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

    def plot_accuracy(history: dict, round_str: str):
        title = 'Accuracy (Training and Validation)'
        maxacc = max(max(history['val_categorical_accuracy']), max(history['categorical_accuracy']))
        plt.plot(history['categorical_accuracy'], label='Training Accuracy',color=result_12_color_sequence[3])
        plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy',color=result_12_color_sequence[2])
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([maxacc-0.35, maxacc+0.05])
        if round_str:
            plt.title(f"{round_str}\n{title}")
        else:
            plt.title(f"{title}")

    def plot_loss(history: dict, round_str: str):
        title = 'Loss (Training and Validation)'
        plt.plot(history['loss'], label='Training Loss',color=result_12_color_sequence[5])
        plt.plot(history['val_loss'], label='Validation Loss',color=result_12_color_sequence[4],)
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        #plt.ylim([0, 1.0])
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
    sns.heatmap(cf, annot=True, xticklabels=labels, yticklabels=labels, cmap=continuous_cmap, vmax=10)
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
    if gradcam : nb= nb*2
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


def compare_models_confusions(active_models: list, save: bool = False, fig_dir : str = '0notdefined') -> None:
    """
    Compare the confusion matrices of multiple models and plot the results.

    Parameters:
    - active_models (list): A list of active models to compare.
    - save (bool, optional): Whether to save the generated plot. Defaults to False.
    - fig_dir (str, optional): The path to save the plot if save is True. Defaults to '0notdefined'.

    Returns:
    None
    """
    data = active_models[0].data
    campaign_id= active_models[0].campaign_id
    confusions_df = pd.DataFrame(columns=['actual', 'count', 'confusion', 'model'])
    max = 0
    for m in active_models:
        cf = confusion_matrix(m.results.actual, m.results.predicted)
        for i in range(cf.shape[0]):
            for j in range(cf.shape[1]):
                if i == j:
                    count = 0
                else:
                    count = cf[i][j]
                if count > max: max = count
                confusions_df.loc[len(confusions_df)] = {
                    'actual': data.classes[i], 'count': count, 'confusion': data.classes[j], 'model': m.record_name
                }
    fig = px.bar(confusions_df, y='count', x='actual', animation_frame='model', color='confusion',
                 color_discrete_sequence=result_12_color_sequence, template=plotly_template)
    fig.update_layout(yaxis=dict(range=[0, max + 2]))
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 3000
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 800
    fig.update_layout(title=f"class confusion evolution between models –– {campaign_id} campaign")
    if save : pio.write_html(fig, f"{fig_dir}/{campaign_id}/compare_confusions.html", auto_open=False)
    else : fig.show()


def compare_models_performances(active_models: list, save: bool = False, fig_dir : str = '0notdefined') -> None:
    """
    Compare the performances of multiple models and plot the results.

    Parameters:
    - active_models (list): A list of active models to compare.
    - save (bool, optional): Whether to save the generated plot. Defaults to False.
    - fig_dir (str, optional): The path to save the plot if save is True. Defaults to '0notdefined'.

    Returns:
    None
    """
    limit_finetuning = None
    campaign_id= active_models[0].campaign_id
    epoch_evolution = pd.DataFrame(columns=['model', 'epochs', 'loss', 'accuracy', 'val'])
    for m in active_models:
        epoch1 = len(m.history1['loss'])
        for i in range(epoch1):
            epoch_evolution.loc[len(epoch_evolution)] = {
                'model': m.record_name, 'epochs': i + 1, 'loss': m.history1['loss'][i],
                'accuracy': m.history1['categorical_accuracy'][i], 'val': ''
            }
            epoch_evolution.loc[len(epoch_evolution)] = {
                'model': m.record_name, 'epochs': i + 1, 'loss': m.history1['val_loss'][i],
                'accuracy': m.history1['val_categorical_accuracy'][i], 'val': 'val'
            }
        if m.history2:
            for i in range(len(m.history2['loss'])):
                epoch_evolution.loc[len(epoch_evolution)] = {
                    'model': m.record_name, 'epochs': i + epoch1 +1, 'loss': m.history2['loss'][i],
                    'accuracy': m.history2['categorical_accuracy'][i], 'val': ''
                }
                epoch_evolution.loc[len(epoch_evolution)] = {
                    'model': m.record_name, 'epochs': i + epoch1 + 1, 'loss': m.history2['val_loss'][i],
                    'accuracy': m.history2['val_categorical_accuracy'][i], 'val': 'val'
                }
            limit_finetuning =epoch1
    for metric in ['loss', 'accuracy']:
        plt.figure(figsize=(20, 10))
        fig = px.line(data_frame=epoch_evolution, x='epochs', y=metric, line_dash='val', color='model',
                      color_discrete_sequence=result_12_color_sequence, template=plotly_template, markers=True)
        fig.update_traces(line={'width': 2})
        if limit_finetuning:
            fig.add_vline(x=limit_finetuning, line_width=2, line_dash="dash", line_color="red")
        fig.update_layout(title=f"{metric} evolution between models –– {campaign_id} campaign")
        if save:
            pio.write_html(fig, f"{fig_dir}/{campaign_id}/compare_performances_{metric}.html", auto_open=False)
        else:
            fig.show()