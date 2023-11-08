
import src.features as lf
import src.models as lm

#### Directories
lm.models.RECORD_DIR='../models/records'
lm.models.FIGURE_DIR='../reports/figures'

from numpy.random import seed
seed(777)
import tensorflow as tf
tf.random.set_seed(777)

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')


import src.models as lm
import src.visualization as lv
import src.features as lf

import tensorflow as tf
from tensorflow import keras

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import *
from keras import Model, Input, Sequential
from keras.regularizers import l2



class TestTraining(lm.models.Trainer):
    # abstract class
    record_name = "none"
    lr2=0.0001
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,  # min_delta=1e-3,
                                     patience=3,  # tf.math.ceil(epoch1/10),
                                     verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=5,  # min_delta=5e-3,#tf.math.ceil(epoch1/5),
                                  restore_best_weights=True, verbose=1)

    training_param = None
    epoch1 = 12
    epoch2 = 20

    def __init__(self, data_wrapper, campaign_id):
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)
        inputs = tf.keras.layers.Input([None, None, 3])
        x= self.base_model.model(inputs, training=self.training_param)
        #x = self.base_model.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        #self.model = Model(inputs=self.base_model.model.input, outputs=output)
        self.model = Model(inputs=inputs, outputs=output)





class T1a(TestTraining):
    record_name = "50-1step-NoNorm"
    training_param = None
    def process_training(self):
        self.make_trainable_base_model_last_layers(50)
        self.compile_fit(lr=self.lr1, epochs=self.epoch1+self.epoch2)

class T1b(TestTraining):
    record_name = "50-1step-Norm"
    training_param = None
    def process_training(self):
        self.make_trainable_base_model_last_layers(50,remove_normalization=False)
        self.compile_fit(lr=self.lr1, epochs=self.epoch1+self.epoch2)


class T1c(TestTraining):
    record_name = "50-1step-Norm-noInference"
    training_param = False
    def process_training(self):
        self.make_trainable_base_model_last_layers(50,remove_normalization=False)
        self.compile_fit(lr=self.lr1, epochs=self.epoch1+self.epoch2)




class T2a(TestTraining):
    record_name = "50-2steps-NoNorm"
    training_param = None
    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

        self.make_trainable_base_model_last_layers(50)
        self.compile_fit(lr=self.lr2, epochs=self.epoch2, is_fine_tuning=True)

class T2b(TestTraining):
    record_name = "50-2steps-Norm"
    training_param = None
    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

        self.make_trainable_base_model_last_layers(50,remove_normalization=False)
        self.compile_fit(lr=self.lr2, epochs=self.epoch2, is_fine_tuning=True)

class T2c(TestTraining):
    record_name = "50-2steps-Norm-noInference"
    training_param = False
    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

        self.make_trainable_base_model_last_layers(50,remove_normalization=False)
        self.compile_fit(lr=self.lr2, epochs=self.epoch2, is_fine_tuning=True)


#### Campaign init
models = [
    T1a,
    T1b,
    T1c,
    T2a,
    T2b,
    T2c,
  ]
campaign= lm.campaign.Campaign(campaign_id='test-finetuning2', data_wrapper=data_wrapper, models=models)

#### train
#### models will be serialized in RECORD_DIR (launch only once by campaign)

#campaign.train_all()


#### evaluate
#### models will be loaded and results saved in FIGURE_DIR/campaign_id

campaign.evaluate_and_build_reports(gradcam=False)