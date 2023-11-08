from sklearn.metrics import classification_report

import matplotlib
import matplotlib.pyplot as plt
import src.models as lm
import src.visualization as lv
import src.features as lf
from skimage.morphology import closing
from skimage.morphology import disk 
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras import Model, Input, Sequential
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, Hinge
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import DataFrameIterator
from keras.utils import img_to_array, load_img

import pickle

""" Model training """

RECORD_DIR: str = "../models/records"
FIGURE_DIR: str = "../reports/figures"

from numpy.random import seed
seed(777)
import tensorflow as tf
tf.random.set_seed(777)


class Trainer():
    #abstract class
    """
    Trainers are classes making model init, training and estimation
    """
    # default parameters


    img_size: tuple = (224, 224)
    data_augmentation = dict(
        validation_split=0.12,
        rotation_range=360,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
    )

    record_name: str = None
    epoch1: int = 12 #12
    lr1: float = 1e-3


    # used for 2 rounds fine tuning
    epoch2: int = 20 #30
    lr2: float = lr1*1e-1

    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, # min_delta=1e-3,
                                     patience=3,#tf.math.ceil(epoch1/10),
                                     verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=10, # min_delta=5e-3,#tf.math.ceil(epoch1/5),
                                  restore_best_weights=True, verbose=1)
    
    batch_size = 32

    # other attributes that will be init by subclasses
    base_model: lm.model_wrapper.BaseModelWrapper = None
    model: Model = None
    data: lf.data_builder.ImageDataWrapper = None
    train: DataFrameIterator = None
    validation: DataFrameIterator = None
    test: DataFrameIterator = None
    history1: dict = None
    history2: dict = None
    results: pd.DataFrame = None
    campaign_id: str = None

    def get_best_models_callback(self):
        _,_,file = self.get_filenames()
        # Merci de ne pas faire ce qui est commenté mais de changer le chemin : lm.models.RECORD_DIR="./models/records"  dans le votre fichier principal
        # file = f"./models/records/test/{self.campaign_id}/{self.record_name}_model.h5"
        return tf.keras.callbacks.\
                ModelCheckpoint(filepath=file,
                        monitor="val_categorical_accuracy", verbose=1, save_best_only=True)


    def __init__(self, data_wrapper, campaign_id="default"):
        self.campaign_id = campaign_id
        self.data = data_wrapper
        # build data flows
        if data_wrapper.train_df is not None:
            self.train, self.validation, self.test = lf.data_builder.get_data_flows(self.data, self.base_model,
                                                                                self.batch_size, self.data_augmentation,
                                                                                self.img_size)

    def __str__(self):
        return self.record_name


    def serialize(self):
        """
        Serializes the model and history objects to disk.

        Parameters:

        timestamp_str (str): Optional timestamp or ID string used to create the directory for storing the serialized files.
        If not provided, the files will be stored in the main record directory.

        Returns:
            List[str]: A list of file paths containing the serialized model and history filenames.
        """
        history1_file, history2_file, model_file = self.get_filenames()
        #self.model.save(model_file)
        with open(history1_file, 'wb') as f:
            pickle.dump(self.history1, f)
        if (self.history2):
            with open(history2_file, 'wb') as f:
                pickle.dump(self.history2, f)
        return [model_file, history1_file, history2_file]

    def save_fig(self, plot: matplotlib.pyplot, name: str):
        path = f"{FIGURE_DIR}/{self.campaign_id}/{self.record_name}_{name}"
        print(f"Saving figure to {path}")
        plot.savefig(path, bbox_inches='tight')
        

    def get_filenames(self):
        path = f"{RECORD_DIR}/{self.campaign_id}/{self.record_name}"
        model_file = path + '_model.h5'
        history1_file = path + '_history1.pkl'
        history2_file = path + '_history2.pkl'
        return history1_file, history2_file, model_file

    def deserialize(self):
        """
        Deserialize the model, history1, and history2 from the given timestamp string.

        :return: None
        """
        history1_file, history2_file, model_file = self.get_filenames()
        # Merci de ne pas faire ce qui est commenté mais de changer le chemin : lm.models.RECORD_DIR="./models/records"  dans le votre fichier principal
        #model_file = f"./models/records/{self.campaign_id}/{self.record_name}_model.h5"
        self.model = keras.models.load_model(model_file)
        print('++++++model+++',end='')
        with open(history1_file, 'rb') as f:
            self.history1 = pickle.load(f)
        print('+++history1+++', end='')
        if os.path.exists(history2_file):
            with open(history2_file, 'rb') as f:
                self.history2 = pickle.load(f)
            print('+++history2+++')
        else: print('')

    def print_step(self, step_name: str) -> str:
        print(f">>> {self.record_name} –– {step_name}")

    def add_background_removal(self):
        """
        Adds a background removal step to the preprocessing pipeline of the base model.

        This function modifies the `preprocessing` attribute of the `base_model` object.
        It replaces the existing preprocessing function with a new function that performs
        background removal on the input image.

        Parameters:
            None

        Returns:
            None
        """
        preproc = self.base_model.preprocessing

        def preprocessing_lambda(image):
            if image.dtype == 'float32':
                norm=True
                image2 = (image - np.min(image)) / (np.max(image) - np.min(image))
            image2 = lf.segmentation.remove_background(image2)
            #TODO : voir avec Iréné si la correction est correcte
            if norm:
                image2 = image2 * (np.max(image) - np.min(image)) + np.min(image)
            return preproc(image2)

        self.base_model.preprocessing = preprocessing_lambda


    def fit_or_load(self, training=True):
        """
        Fit or load the model for training or inference.
        Parameters:
            training (bool): A flag indicating whether to perform training or inference.
        Returns:
            None
        """
        if (training):
            self.print_step("Training")
            self.process_training()
            self.print_step("Serialize ")
            self.serialize()
        else:
            self.print_step("Loading")
            self.deserialize()

    def make_trainable_base_model_last_layers(self, layer_percent: int = 10, remove_normalization :bool = True):
        """
        Makes the last `layer_percent` layers of the base model trainable.
        Avoid making trainable the Normalization layers
        Args:
            layer_percent (int): The percent of last layers to make trainable. Defaults to 10.

        Returns:
            None
        """
        self.base_model.model.trainable = True
        nb_layers = round(layer_percent / 100 * len(self.base_model.model.layers))
        print(f"train last {nb_layers} layers")
        
        for layer in self.base_model.model.layers:
            if (remove_normalization & ('Normalization' in str(type(layer)))):
                layer.trainable = False
            else:
                layer.trainable = True
        
        for layer in self.base_model.model.layers[:-nb_layers]:
            layer.trainable = False


    def process_training(self):
        """
        implement the training sequence using compile_fit method
        must be implemented by subclasses
        """
        raise NotImplementedError("process_training must be implemented")   

    def compile_fit(self, lr: float, epochs: int, is_fine_tuning: bool = False):
        """
                Train the model and keep trace of history
                The model can then be serialized
                Parameters:
                    epochs (int): The number of epochs to train the model for
                Returns:
                    None
        """
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=CategoricalCrossentropy(),
            metrics=CategoricalAccuracy()
        )

        history = self.model.fit(
            self.train,
            validation_data=self.validation,
            epochs=epochs,
            batch_size=self.batch_size,
            class_weight=self.data.weights,
            callbacks=[self.lr_reduction, self.stop_callback, self.get_best_models_callback()]
        ).history

        if (is_fine_tuning) : self.history2 = history
        else : self.history1 = history


    def single_prediction(self, path: str) -> str:
        """
        Evaluate the model on a single image.
        Parameters:
            path (str): The path to the image.
        Returns:
            str: The predicted class
        """
        self.print_step("Evaluation")
        img = load_img(path, target_size=self.img_size)
        return self.predict_image(img)

    def predict_image(self, img : np.ndarray) -> str:
        img = img_to_array(img)
        img = self.base_model.preprocessing(img)
        pred_vector = self.model.predict(np.expand_dims(img, axis=0))
        return self.data.classes[np.argmax(pred_vector)]




    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate the model by getting predictions and storing them in the results attribute.
        Parameters:
            None (it uses the test flow encapsultated by the Trainer
        Returns:
            pd.DataFrame: The results dataframe
        """
        self.print_step("Evaluation")
        self.results = lf.data_builder.get_predictions_dataframe(self.model, self.test, self.data.test_df)
        return self.results

    def save_evaluation_reports(self, gradcam)-> None :
        """
        Save all the reports in FIGURE_DIR and return it as a string.
        """
        if( not self.results) : self.evaluate()
        self.print_classification_report( save=True)
        self.display_history_graphs( save=True)
        self.display_confusion_matrix(save=True)
        self.display_samples(nb=3,save=True)
        if gradcam :
            self.display_samples(nb=12, gradcam=True, guidedGrad_cam=True, segmented=False,  save=True)
            self.display_samples(nb=12, gradcam=True, guidedGrad_cam=True, segmented=True,  save=True)

    def print_classification_report(self, save=False) -> str:
        """
        Print the classification report and return it as a string.
        """
        self.print_step("Classification Report")
        cr = classification_report(self.results.actual, self.results.predicted)
        if (save) :
            fig, ax = plt.subplots(figsize=(6, 2))
            font = plt.rcParams["font.family"]
            plt.rcParams["font.family"] = "monospace"
            ax.axis('off')
            titled_cr = f"                   {self.record_name} – Classification Report\n\n{cr}"
            ax.text(0, 2, titled_cr, horizontalalignment='left', verticalalignment='center',
                    fontsize=11, transform=ax.transAxes)
            self.save_fig(plt,"classification_report.png")
            plt.rcParams["font.family"] = font
        else: print(cr)
        return cr

    def display_history_graphs(self, save=False) -> None:
        """
        Display the graphs of the training and validation history.
        """
        self.print_step("Display training history graphs")
        plot = lv.graphs.plot_history_graph(self.history1, self.history2, self.record_name, show=(not save))
        if save: self.save_fig(plot, "history_graph.png")


    def display_samples(self, nb: int, include_true_pred = True, 
                        include_false_pred = True, gradcam: bool = False, 
                        segmented:bool=False, guidedGrad_cam : bool =False, save=False) -> None :
        """
        Display training data samples with gradcam if gradcam is True.

        Parameters:
            nb (int): The number of samples to display.
            include_true_pred (bool, optional): Whether to include samples with true predictions. Defaults to True.
            include_false_pred (bool, optional): Whether to include samples with false predictions. Defaults to True.
            gradcam (bool, optional): Whether to show GradCAM layer or not. Defaults to False.

        Returns:
            None
        """
        self.print_step("Display training data samples")
        results = self.results
        if (include_true_pred and not include_false_pred): results = results[results['Same']==True]
        if (not include_true_pred and  include_false_pred): results = results[results['Same']==False]
        if gradcam :
            plot = lv.graphs.display_results(results, nb=nb, record_name=self.record_name, gradcam=True, model=self.model,
                        base_model_wrapper=self.base_model, img_size=self.img_size, segmented=segmented, guidedGrad_cam=guidedGrad_cam, show=(not save))
        else : plot = lv.graphs.display_results(results, nb=nb, record_name=self.record_name, show=(not save))
        if (save) :
            endstr = ''
            if segmented: endstr += '_segmented'
            if gradcam : endstr += '_gradcam'
            if guidedGrad_cam: endstr += '_guidedGradcam'
            self.save_fig(plot, f"sample_graphs{endstr}.png")

    def display_confusion_matrix(self, save = False) -> None:
        """
       This function display the confusion matrix for the results of the model.
       It uses the `plot_confusion_matrix` function from the `lv.graphs` module to plot the matrix.

        Parameters:
        - None

        Returns:
        - None
        """
        self.print_step("Display confusion matrix")
        plot = lv.graphs.plot_confusion_matrix(self.results, self.record_name, show = (not save))
        if (save) :
            self.save_fig(plot, f"confusion_matrix.png")