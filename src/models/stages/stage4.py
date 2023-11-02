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
        super().__init__(data_wrapper, campaign_id)
        x = self.base_model.model.output
        # x = Conv2D(128, (6, 6), activation='relu', padding='same', kernel_regularizer= l2(0.001))(x)
        # x = BatchNormalization()(x)
        # x = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        # x = BatchNormalization()(x)
        # x = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling2D(2, 2)(x)
        # x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        # x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        # x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)        
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

        self.make_trainable_base_model_last_layers(50)
        self.compile_fit(lr=self.lr2, epochs=self.epoch2, is_fine_tuning=True)



class Stage4MobileNetv3(Stage4):
    record_name = "Stage-4_MobileNetv3"

    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)

class Stage4ResNetv2(Stage4):
    record_name = "Stage-4_ResNetv2"

    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size)
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)
