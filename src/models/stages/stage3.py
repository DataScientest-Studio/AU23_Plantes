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

    def __init__(self, data_wrapper, campaign_id):
        super().__init__(data_wrapper, campaign_id)
        x = self.base_model.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

        self.make_trainable_base_model_last_layers(10)
        self.compile_fit(lr=self.lr2, epochs=self.epoch2, is_fine_tuning=True)



class Stage3MobileNetv3(Stage3):
    record_name = "Stage-3_MobileNetv3"

    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)

class Stage3ResNetv2(Stage3):
    record_name = "Stage-3_ResNetv2"

    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size)
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)
