import src.models as lm
import src.visualization as lv
import src.features as lf

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, AveragePooling2D, Dropout
from keras import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, Hinge
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pickle

""" Model training """



class Trainer() :
    """
    Trainers are classes making model init, training and estimation
    """
    # default parameters
    record_dir = "../../models/records/"
    img_size = 224 ## TODO change for tuple
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
    epoch1=1
    epoch2=1
    batch_size=32

    # other attributes that will be init by subclasses
    parameters = None
    model_wrapper = None
    data_wrapper = None
    train = None
    validation = None
    test = None
    base_model = None
    model = None
    history1 = None
    history2 = None
    record_name = None
    results= None


    def serialize(self, timestamp_str=None):
        """
        Serializes the model and history objects to disk.

        Parameters:

        timestamp_str (str): Optional timestamp or ID string used to create the directory for storing the serialized files.
        If not provided, the files will be stored in the main record directory.

        Returns:
            List[str]: A list of file paths containing the serialized model and history filenames.
        """
        history1_file, history2_file, model_file = self.get_filename(timestamp_str)
        self.model.save(model_file)
        with open(history1_file, 'wb') as f:
            pickle.dump(self.history1, f)
        if (self.history2):
            with open(history2_file, 'wb') as f:
                pickle.dump(self.history2, f)
        return [model_file, history1_file, history2_file]


    def get_filename(self, timestamp_str):
        if (timestamp_str):
            path = f"{self.record_dir}/{timestamp_str}/{self.record_name}"
        else:
            path = f"{self.record_dir}/{self.record_name}"
        model_file = path + '_model.h5'
        history1_file = path + '_history1.pkl'
        history2_file = path + '_history2.pkl'
        return history1_file, history2_file, model_file


    def deserialize(self, timestamp_str):
        """
        Deserialize the model, history1, and history2 from the given timestamp string.

        :param timestamp_str: The timestamp string representing the desired snapshot.
        :type timestamp_str: str

        :return: None
        """
        history1_file, history2_file, model_file = self.get_filename(timestamp_str)
        self.model = keras.models.load_model(model_file)
        with open(history1_file, 'rb') as f:
            self.history1 = pickle.load(f)
        if (self.history2):
            with open(history2_file, 'rb') as f:
                self.history2 = pickle.load(f)

    """
        Train the model and keep trace of history
        The model can then be serialized
        Parameters:
            epochs (int): The number of epochs to train the model for
        Returns:
            None
    """
    def train(self, epochs:int):

        # pretrained model is set not trainable
        self.base_model.trainable = False

        self.model.compile(
            optimizer=Adam(),
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


    """
    Evaluate the model by getting predictions and storing them in the results attribute.
    Parameters:
        None
    Returns:
        None
    """
    def evaluate(self):
        self.results = lf.data_builder.get_predictions_dataframe(self.model, self.test, self.data_wrapper.test)


### STEP 1 : simple training (no fine tuning, no background removal, simple classification


class Step1MobileNetv3(Trainer) :
    record_name = "step1_mobilenetv3"
    def __init__(self, data_wrapper):
        self.model_wrapper = lm.model_wrapper.MobileNetv3(self.img_size)
        self.data_wrapper = data_wrapper

        # build data flows
        self.train, self.validation, self.test = lf.data_builder.get_data_flows(self.data_wrapper, self.model_wrapper, self.batch_size, self.data_augmentation, self.img_size)


        ##TODO virer gradcam
        gradcam_encapsulation = lm.model_wrapper.get_gradcam_encapsulation(self.model_wrapper)
        base_model = self.model_wrapper.model

        # build model
        x, gradcam_output = gradcam_encapsulation(base_model.input, training=False)
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(224, activation='leaky_relu')(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=base_model.input, outputs=[output, gradcam_output])

    """
        Train the model and keep trace of history
        The model can then be serialized
        Parameters:
            None
        Returns:
            None
    """
    def train(self):
        print ('cxoucou')
        self.base_model.trainable = False
        super.train(self, epochs=self.epoch1)





