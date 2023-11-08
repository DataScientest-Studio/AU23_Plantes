from typing import Tuple
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.cm as cm
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
from keras.utils import img_to_array, load_img
from keras import Model

import pandas as pd
from types import SimpleNamespace
import numpy as np
import src.models as lm
import cv2 

from src.features import segmentation
# data_dir= 'sources/nonsegmentedv2/'
data_dir:str = 'sources/v2-plant-seedlings-dataset/'

random_state:int = 42


class ImageDataWrapper() :
    """
    Wrapper for encapsulating a splited image data frame and its characteritics
    it holds the following attributes
        - data : the complete dataframe
        - train : the train dataset
        - test : the test dataset
        - classes : the list of classes
        - nb_classes : the number of classes
        - weights : the class weights
    """
    dataframe = None
    train_df = None
    test_df = None
    classes = None
    nb_classes = None
    weights = None


    def __init__(self, dataframe: pd.DataFrame, train_df : pd.DataFrame, test_df: pd.DataFrame, classes: list, nb_classes:int, weights:dict) -> None:
        self.dataframe = dataframe
        self.train_df = train_df
        self.test_df = test_df
        self.classes = classes
        self.nb_classes = nb_classes
        self.weights = weights

class VoidSeedlingImageDataWrapper(ImageDataWrapper) :
    def __init__(self) :
        self.classes =[
         'Black-grass',
         'Charlock',
         'Cleavers',
         'Common Chickweed',
         'Common wheat',
         'Fat Hen',
         'Loose Silky-bent',
         'Maize',
         'Scentless Mayweed',
         'Shepherd’s Purse',
         'Small-flowered Cranesbill',
         'Sugar beet'
        ]
        self.nb_classes = len(self.classes)
        self.dataframe = None
        self.train_df = None
        self.test_df = None
        self.weights = None


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

    images = pd.Series(images, name="path").astype(str)
    labels = pd.Series(labels, name="label").astype(str)

    data = pd.concat([images, labels], axis=1)

    # Remove bad images
    index_to_drop = data[
        ((data['label'] == 'Sugar beet') & (data['path'].str.endswith('97.png'))) |
        ((data['label'] == 'Common Chickweed') & (data['path'].str.endswith('351.png'))) |
        ((data['label'] == 'Loose Silky-bent') & (data['path'].str.endswith('304.png')))
    ].index
    data.drop(index_to_drop, inplace=True)

    train_df, test_df = train_test_split(data, train_size=train_size, shuffle=shuffle, random_state=random_state)
    classes = sorted(data.label.unique())
    nb_classes = len(classes)

    train_classes = train_df['label'].values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_classes), y=train_classes)
    weights = {i: w for i, w in enumerate(class_weights)}
    
    return ImageDataWrapper(dataframe=data, train_df=train_df, test_df=test_df, classes= classes, nb_classes=nb_classes, weights=weights)

def get_data_flows(image_data_wrapper: ImageDataWrapper, base_model: lm.model_wrapper.BaseModelWrapper, batch_size: int, data_augmentation: dict, img_size : tuple) -> Tuple :
    """
    Generate data flows for training and testing a model.

    Parameters:
        base_model (ModelWrapper): An instance of the ModelWrapper class.
        parameters (dict): A dictionary containing various parameters for data generation.
        image_data_wrapper (ImageDataWrapper): A DataFrame containing the training data.

    Returns:
        A tuple with the train, validation, and test data generators.
    """
    train_generator = ImageDataGenerator(
        preprocessing_function=base_model.preprocessing,
        **data_augmentation,
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=base_model.preprocessing,
    )

    generator_param = dict(
        x_col="path",
        y_col="label",
        target_size=img_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        seed=random_state,
        verbose=False,
    )

    train = train_generator.flow_from_dataframe(
        dataframe=image_data_wrapper.train_df,
        shuffle=True,
        subset='training',
        **generator_param
    )

    validation = train_generator.flow_from_dataframe(
        dataframe=image_data_wrapper.train_df,
        shuffle=False,
        subset='validation',
        **generator_param
    )

    test = test_generator.flow_from_dataframe(
        dataframe=image_data_wrapper.test_df,
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
    predictions = np.argmax(model.predict(test_flow), axis=1)
    # generate a dataframe with the predicted and real labels
    result_df = test_df.copy()
    result_df = result_df.rename(columns={
        'path': 'filename',
        'label': 'actual'
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



def last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'Conv' in str(type(layer)) :
            return layer
    print("couche gradcam non trouvée - return last layer")
    return model.layers[-1]


def make_gradcam_heatmap(img_array: np.ndarray, complete_model : Model, base_model_wrapper : lm.model_wrapper.BaseModelWrapper) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap for a given input image array using a model.

    Args:
        img_array (numpy.ndarray): The input image array.
        model (tensorflow.keras.Model): The model to generate the heatmap from.
        base_model_wrapper (lm.model_wrapper.BaseModelWrapper): The associated base model wrapper.

    Returns:
        numpy.ndarray: The Grad-CAM heatmap.
    """

    with tf.GradientTape() as tape:
        img_array = base_model_wrapper.preprocessing(img_array)
        grad_model = Model(complete_model.input, [complete_model.output, last_conv_layer(complete_model).output])
        #grad_model = Model(complete_model.input, [complete_model.output, complete_model.get_layer(base_model_wrapper.grad_cam_layer).output])
        preds, last_conv_layer_output = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def gradCAMImage(img_path :str, img_size : tuple, model : Model, 
                base_model_wrapper : lm.model_wrapper.BaseModelWrapper, 
                segmented:bool=True, guidedGrad_cam : bool =False) -> np.ndarray:
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
    path = f"{img_path}"
    img = load_img(path, target_size=img_size)
    img = img_to_array(img)
    expended_img = np.expand_dims(img / 255, axis=0)
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
        img = resize(plt.imread(path), output_shape=img_size)
        img = segmentation.remove_background(x=img)
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

def guided_backprop(model, img : np.ndarray, upsample_size: tuple[int, int]):
        import cv2 

        
        base_model           = Model(model.input, [model.output, last_conv_layer(model).output])
        base_model.trainable = True
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(img, tf.float32)
            tape.watch(inputs)
            outputs = base_model(inputs)

        grads       = tape.gradient(outputs, inputs)[0]
        saliency    = cv2.resize(np.asarray(grads), upsample_size)

        return saliency

def deprocess_image(x : np.ndarray):
    import keras.backend as K
    
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def overlay_gradCAM(img, cam3):
    import cv2

    cam3    = np.uint8(255*cam3)
    cam3    = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    new_img = 0.3*cam3 + 0.5*img
    
    return (new_img*255.0/new_img.max()).astype("uint8")
