from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import src.models as lm

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras import Model, Input, Sequential

"""



    Stage 1 : simple training (no fine tuning, no background removal, simple classification)
    comparison between the 2 selected models : Mobilnetv3 and Resnet


"""


class Stage1(lm.models.Trainer):
    # abstract class

    img_size: tuple = (224, 224)
    data_augmentation = dict(
        validation_split=0.12,
        rotation_range=360,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
    )
    epoch1: int = 32
    lr1: float = 1e-3

    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                     patience=3,
                                     verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=5,
                                  restore_best_weights=True,
                                  verbose=1)

    batch_size = 32

    def __init__(self, data_wrapper: lm.model_wrapper.BaseModelWrapper, campaign_id: str):
        super().__init__(data_wrapper,campaign_id)
        self.model_definition()

    def model_definition(self):
        x = self.base_model.model.output
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)



class MobileNetv3(Stage1):
    record_name = "1-simple-Mob"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size, pooling='avg')
        super().__init__(data_wrapper, campaign_id)


class ResNetv2(Stage1):
    record_name = "1-simple-Res"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size, pooling='avg')
        super().__init__(data_wrapper, campaign_id)


