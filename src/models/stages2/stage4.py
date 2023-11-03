import src.models as lm
import src.visualization as lv
import src.features as lf

import tensorflow as tf
from tensorflow import keras

from keras.layers import *
from keras import Model, Input, Sequential
from keras.regularizers import l2



"""

    Stage 4 : advanced  classification + fine-tuning with background removal
    comparison between the 2 selected models : Mobilnetv3 and Resnet


"""

class Stage4(lm.models.Trainer):
    # abstract class
    record_name = "none"

    def __init__(self, data_wrapper, campaign_id):
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)
        x = self.base_model.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
        self.make_trainable_base_model_last_layers(50, remove_normalization=False)
        self.compile_fit(lr=self.lr1, epochs=self.epoch1+self.epoch2)



class CNN(Stage4):
    record_name = "4-dense-Cnn"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.SimpleCNN(self.img_size)
        super().__init__(data_wrapper, campaign_id)


class MobileNetv3(Stage4):
    record_name = "4-dense-Mob"

    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        super().__init__(data_wrapper, campaign_id)

class ResNetv2(Stage4):
    record_name = "4-dense-Res"

    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size)
        super().__init__(data_wrapper, campaign_id)
