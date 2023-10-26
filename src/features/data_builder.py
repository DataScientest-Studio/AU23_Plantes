from typing import Tuple

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.cm as cm

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
from keras.utils import img_to_array, load_img
from keras import Model

import pandas as pd
from types import SimpleNamespace
import numpy as np
import src.models as lm


# data_dir= 'sources/nonsegmentedv2/'
data_dir:str = 'sources/v2-plant-seedlings-dataset/'

random_state:int = 42


class ImageDataWrapper() :
    """
    Wrapper for encapsulating a splited image data frame and its characteritics
    it holds the following attributes
        - train : the train dataset
        - test : the test dataset
        - classes : the list of classes
        - nb_classes : the number of classes
        - weights : the class weights
    """
    train = None
    test = None
    classes = None
    nb_classes = None
    weights = None


    def __init__(self, train : pd.DataFrame, test: pd.DataFrame, classes: list, nb_classes:int, weights:dict) -> None:
        self.train = train
        self.test = test
        self.classes = classes
        self.nb_classes = nb_classes
        self.weights = weights



def create_dataset_from_directory (data_dir: str = data_dir,  train_size: float = 0.9, shuffle: bool = True,) -> ImageDataWrapper:
    """
    Creates a dataset from the given data directory.
    Args:
        data_dir (str): The path to the data directory. Defaults to the `data_dir` variable.
        shuffle (bool): Whether to shuffle the dataset. Defaults to `True`.
        train_size (float): The proportion of the dataset to use for training. Defaults to 0.9.
    Returns:
        ImageDataWrapper: The dataset.
    """
    dataset_path = Path(data_dir)
    images = list(dataset_path.glob(r'**/*.png'))
    labels = list(map(lambda x: x.parents[0].stem, images))

    images = pd.Series(images, name="Images").astype(str)
    labels = pd.Series(labels, name="Labels").astype(str)

    data = pd.concat([images, labels], axis=1)

    train_df, test_df = train_test_split(data, train_size=train_size, shuffle=shuffle, random_state=random_state)
    classes = sorted(data.Labels.unique())
    nb_classes = len(classes)

    train_classes = train_df['Labels'].values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_classes), y=train_classes)
    weights = {i: w for i, w in enumerate(class_weights)}

    return ImageDataWrapper(train=train_df, test=test_df, classes= classes, nb_classes=nb_classes, weights=weights)



def get_data_flows( image_data_wrapper: ImageDataWrapper, model_wrapper: lm.model_wrapper.ModelWrapper, batch_size: int , data_augmentation: dict, img_size : tuple) -> Tuple :
    """
    Generate data flows for training and testing a model.

    Parameters:
        model_wrapper (ModelWrapper): An instance of the ModelWrapper class.
        parameters (dict): A dictionary containing various parameters for data generation.
        image_data_wrapper (ImageDataWrapper): A DataFrame containing the training data.

    Returns:
        A tuple with the train, validation, and test data generators.
    """
    train_generator = ImageDataGenerator(
        preprocessing_function=model_wrapper.preprocessing,
        **data_augmentation,
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=model_wrapper.preprocessing,
    )

    generator_param = dict(
        x_col="Images",
        y_col="Labels",
        target_size=img_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        seed=random_state,
    )

    train = train_generator.flow_from_dataframe(
        dataframe=image_data_wrapper.train,
        shuffle=True,
        subset='training',
        **generator_param
    )

    validation = train_generator.flow_from_dataframe(
        dataframe=image_data_wrapper.train,
        shuffle=True,
        subset='validation',
        **generator_param
    )

    test = test_generator.flow_from_dataframe(
        dataframe=image_data_wrapper.test,
        shuffle=False,
        **generator_param
    )

    return train, validation, test


def get_predictions_dataframe(model : Model, test_flow : DataFrameIterator , test_df : pd.DataFrame, shuffle:bool=True, random_state:int=random_state) -> pd.DataFrame:
    """
    Generate a dataframe with predicted and actual labels based on the given model and test data.
    This dataframe has 4 columns :
    - filename: the name of the image
    - predicted: the predicted label
    - actual: the actual label
    - same: a boolean indicating whether the predicted and actual labels are the same

    Parameters:
    model (object): The trained model used for prediction.
    test_flow (array-like): The test data flow.
    test_df (object): The test dataframe
    shuffle (bool, optional): Whether to shuffle the dataframe. Defaults to True.

    Returns:
        DataFrame: A dataframe containing the predicted and actual labels, as well as a column indicating whether they are the same.
    """
    predictions = np.argmax(model.predict(test_flow)[0], axis=1)
    # generate a dataframe with the predicted and real labels
    result_df = test_df.copy()
    result_df = result_df.rename(columns={
        'Images': 'filename',
        'Labels': 'actual'
    })
    label_dict = dict((v, k) for k, v in test_flow.class_indices.items())
    result_df['predicted'] = [label_dict[i] for i in predictions]
    result_df['Same'] = False
    result_df.loc[result_df['actual'] == result_df['predicted'], 'Same'] = True
    # we shuffle the dataframe
    if (shuffle):
        result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result_df


def readImage(path:str, size:tuple=None) -> np.ndarray:
    """
    Reads an image from the given path and performs some preprocessing steps.
    Parameters:
        path (str): The path to the image file.
    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array.
    """
    if (size) :
        img = load_img(path,color_mode='rgb',target_size=size)
    else :
        img = load_img(path,color_mode='rgb')
    img = img_to_array(img)
    img = img/255.
    return img



def get_single_prediction(model: Model, img_path :str) -> np.ndarray:
    """
    Gets a prediction for an image with the given model.
    """
    img = readImage(img_path)
    return model.predict(np.expand_dims(img, axis=0))[0]




def make_gradcam_heatmap(img_array: np.ndarray, model : Model) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap for a given input image array using a model.

    Args:
        img_array (numpy.ndarray): The input image array.
        model (tensorflow.keras.Model): The model to generate the heatmap from.

    Returns:
        numpy.ndarray: The Grad-CAM heatmap.
    """
    with tf.GradientTape() as tape:
        preds, last_conv_layer_output = model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def gradCAMImage(model : Model, img_path :str, img_size : tuple) -> np.ndarray:
    """
    Generates a Grad-CAM image by overlaying the heatmap on the original image.

    Args:
        model: The neural network model used for generating the heatmap.
        img_path: The path to the input image.
        img_size: The desired size of the input image.

    Returns:
        The superimposed image with the heatmap.
    """
    path = f"{img_path}"
    img = load_img(path, target_size=img_size)
    img = img_to_array(img)
    expended_img = np.expand_dims(img / 255, axis=0)
    heatmap = make_gradcam_heatmap(expended_img, model)
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

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.8 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img