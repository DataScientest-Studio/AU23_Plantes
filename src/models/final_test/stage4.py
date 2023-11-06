import src.models as lm
import src.visualization as lv
import src.features as lf

import tensorflow as tf
from tensorflow import keras

from keras.layers import *
from keras import Model, Input, Sequential
from keras.regularizers import l2
from src.models.final_test.stage3 import Stage3
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""

    Stage 4 : advanced  classification + fine-tuning with background removal
    comparison between the 2 selected models : Mobilnetv3 and Resnet


"""

class Stage4(Stage3):
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                     patience=5,
                                     verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=10,
                                  restore_best_weights=True,
                                  verbose=1)

    l2_param = l2(1e-4)

    def model_definition(self):
        x = self.base_model.model.output
        x = Dense(512, 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, 'relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', kernel_regularizer=self.l2_param, name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)


class MobileNetv3(Stage4):
    record_name = "4-dense-Mob"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size,pooling='avg')
        super().__init__(data_wrapper, campaign_id)

class ResNetv2(Stage4):
    record_name = "4-dense-Res"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size,pooling='avg')
        super().__init__(data_wrapper, campaign_id)
