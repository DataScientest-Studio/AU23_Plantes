import src.models as lm

import tensorflow as tf
from tensorflow import keras

import src.visualization as lv
import src.features as lf

from keras.layers import Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras import Model, Input, Sequential

from src.models.final_test.stage2 import Stage2

"""

    Stage 3 : simple  classification + fine-tuning with background removal
    comparison between the 2 selected models : Mobilnetv3 and Resnet


"""

class Stage3(Stage2):


    def process_training(self):
        self.make_trainable_base_model_last_layers(50, remove_normalization=False)
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)


class MobileNetv3(Stage3):
    record_name = "3-finetuning-Mob"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size,pooling='avg')
        super().__init__(data_wrapper, campaign_id)


class ResNetv2(Stage3):
    record_name = "3-finetuning-Res"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size,pooling='avg')
        super().__init__(data_wrapper, campaign_id)
