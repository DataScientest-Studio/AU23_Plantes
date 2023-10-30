import src.models as lm

import tensorflow as tf
from tensorflow import keras

import src.visualization as lv
import src.features as lf

from keras.layers import Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras import Model, Input, Sequential


"""

    Stage 3 : simple  classification + fine-tuning with background removal
    comparison between the 2 selected models : Mobilnetv3 and Resnet


"""

class Stage3(lm.models.Trainer):
    # abstract class
    record_name = "none"

    def __init__(self, data_wrapper):
        super().__init__(data_wrapper)
        x = self.base_model.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

        self.make_trainable_base_model_last_layers(10)
        self.compile_fit(lr=self.lr2, epochs=self.epoch2)



class Stage3MobileNetv3(Stage3):
    record_name = "Stage-3_MobileNetv3"

    def __init__(self, data_wrapper):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        preprocessing = self.base_model.preprocessing
        def preprocessing_lambda(x):
            return preprocessing(lf.segmentation.remove_background(x))
        self.base_model.preprocessing = preprocessing_lambda
        super().__init__(data_wrapper)

class Stage3ResNetv2(Stage3):
    record_name = "Stage-3_ResNetv2"

    def __init__(self, data_wrapper):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size)
        preprocessing = self.base_model.preprocessing
        def preprocessing_lambda(x):
            return preprocessing(lf.segmentation.remove_background(x))
        self.base_model.preprocessing = preprocessing_lambda
        super().__init__(data_wrapper)
