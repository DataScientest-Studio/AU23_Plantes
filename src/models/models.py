from sklearn.metrics import classification_report

import src.models as lm
import src.visualization as lv
import src.features as lf

import pandas as pd

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, AveragePooling2D, Dropout
from keras import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, Hinge
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import DataFrameIterator

import pickle

""" Model training """


class Trainer():
    """
    Trainers are classes making model init, training and estimation
    """
    # default parameters
    record_dir: str = "../models/records"

    img_size: tuple = (224, 224)
    data_augmentation = dict(
        validation_split=0.12,
        rotation_range=360,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
    )
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                     patience=2, verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=3,
                                  restore_best_weights=True, verbose=1)
    epoch1: int = 1
    lr1: float = 0.001
    # used for 2 rounds fine tuning
    epoch2: int = 1
    lr2: float = 0.00001

    batch_size = 32

    # other attributes that will be init by subclasses
    record_name: str = None
    model_wrapper: lm.model_wrapper.ModelWrapper = None
    data_wrapper: lf.data_builder.ImageDataWrapper = None
    train: DataFrameIterator = None
    validation: DataFrameIterator = None
    test: DataFrameIterator = None
    base_model: Model = None
    model: Model = None
    history1: dict = None
    history2: dict = None
    results: pd.DataFrame = None

    def serialize(self, campain_id=None):
        """
        Serializes the model and history objects to disk.

        Parameters:

        timestamp_str (str): Optional timestamp or ID string used to create the directory for storing the serialized files.
        If not provided, the files will be stored in the main record directory.

        Returns:
            List[str]: A list of file paths containing the serialized model and history filenames.
        """
        history1_file, history2_file, model_file = self.get_filenames(campain_id)
        self.model.save(model_file)
        with open(history1_file, 'wb') as f:
            pickle.dump(self.history1, f)
        if (self.history2):
            with open(history2_file, 'wb') as f:
                pickle.dump(self.history2, f)
        return [model_file, history1_file, history2_file]

    def get_filenames(self, campain_id):
        if (campain_id):
            path = f"{self.record_dir}/{campain_id}/{self.record_name}"
        else:
            path = f"{self.record_dir}/{self.record_name}"
        model_file = path + '_model.h5'
        history1_file = path + '_history1.pkl'
        history2_file = path + '_history2.pkl'
        return history1_file, history2_file, model_file

    def deserialize(self, campaign_id):
        """
        Deserialize the model, history1, and history2 from the given timestamp string.

        :param campaign_id: The timestamp string representing the desired snapshot.
        :type campaign_id: str

        :return: None
        """
        history1_file, history2_file, model_file = self.get_filenames(campaign_id)
        self.model = keras.models.load_model(model_file)
        with open(history1_file, 'rb') as f:
            self.history1 = pickle.load(f)
        if (self.history2):
            with open(history2_file, 'rb') as f:
                self.history2 = pickle.load(f)

    def print_step(self, step_name: str) -> str:
        print(f">>> {self.record_name} –– {step_name}")


    def compile_fit(self, lr: float, epochs: int):
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
            loss=dict(main=CategoricalCrossentropy(), gradcam=None),
            metrics=dict(main=CategoricalAccuracy(), gradcam=None)
        )

        self.history1 = self.model.fit(
            self.train,
            validation_data=self.validation,
            epochs=epochs,
            batch_size=self.batch_size,
            class_weight=self.data_wrapper.weights,
            callbacks=[self.lr_reduction, self.stop_callback]
        ).history



    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate the model by getting predictions and storing them in the results attribute.
        Parameters:
            None
        Returns:
            None
        """
        self.print_step("Evaluation")
        self.results = lf.data_builder.get_predictions_dataframe(self.model, self.test, self.data_wrapper.test)
        return self.results


    def print_classification_report(self) -> str:
        """
        Print the classification report and return it as a string.
        """
        self.print_step("Classification Report")
        cr = classification_report(self.results.actual, self.results.predicted)
        print(cr)
        return cr

    def display_history_graphs(self) -> None:
        """
        Display the graphs of the training and validation history.
        """
        self.print_step("Display training history graphs")
        lv.graphs.plot_history_graph(self.history1, self.history2, self.record_name)

    def display_samples(self, nb: int, include_true_pred = True, include_false_pred = True, gradcam: bool = False) -> None :
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
            lv.graphs.display_results(results, nb=nb, record_name=self.record_name, gradcam=True, model=self.model, img_size=self.img_size)
        else : lv.graphs.display_results(results, nb=nb, record_name=self.record_name)

    def display_confusion_matrix(self) -> None:
        """
       This function display the confusion matrix for the results of the model.
       It uses the `plot_confusion_matrix` function from the `lv.graphs` module to plot the matrix.

        Parameters:
        - None

        Returns:
        - None
        """
        self.print_step("Display confusion matrix")
        lv.graphs.plot_confusion_matrix(self.results, self.record_name)






""""



    Stage 1 : simple training (no fine tuning, no background removal, simple classification)
    comparison between the 2 selected models : Mobilnetv3 and Resnet




"""

class Stage1MobileNetv3(Trainer):
    record_name = "Stage-1_MobileNetv3"

    def __init__(self, data_wrapper):
        self.data_wrapper = data_wrapper

        # set the base model
        self.model_wrapper = lm.model_wrapper.MobileNetv3(self.img_size)
        self.base_model = self.model_wrapper.model

        # build data flows
        self.train, self.validation, self.test = lf.data_builder.get_data_flows(self.data_wrapper, self.model_wrapper,
                                                                                self.batch_size, self.data_augmentation,
                                                                                self.img_size)

        # encapsulation for facilating gradcam computation
        ##TODO virer gradcam
        gradcam_encapsulation = lm.model_wrapper.get_gradcam_encapsulation(self.model_wrapper)

        # Model definition
        x, gradcam_output = gradcam_encapsulation(self.base_model.input, training=False)
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(224, activation='leaky_relu')(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.input, outputs=[output, gradcam_output])



    def fit_or_load(self, campain_id=None, training=True):
        """
        Fit or load the model for training or inference.
        Parameters:
            campain_id (optional): The ID of the campaign to load or serialize.
            training (bool): A flag indicating whether to perform training or inference.
        Returns:
            None
        """
        if (training):
            self.print_step("Training")
            self.base_model.trainable = False
            self.compile_fit(lr=self.lr1, epochs=self.epoch1)

            self.print_step("Serialize ")
            self.serialize(campain_id=campain_id)
        else:
            self.print_step("Loading")
            self.deserialize(campaign_id=campain_id)
