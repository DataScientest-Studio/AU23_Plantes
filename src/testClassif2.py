
import src.features as lf
import src.models as lm
from src.models.stages import *
import src.models as lm
import src.visualization as lv
import src.features as lf

import tensorflow as tf
from tensorflow import keras

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import *
from keras import Model, Input, Sequential
from keras.regularizers import l2


######set the seeds
from numpy.random import seed
seed(777)
import tensorflow as tf
tf.random.set_seed(777)

#### Directories
lm.models.RECORD_DIR='../models/records'
lm.models.FIGURE_DIR='../reports/figures'

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')






class TestClassif(lm.models.Trainer):
    # abstract class
    record_name = "none"
    lr1=0.001
    epoch1 = 32
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                     patience=5,
                                     verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=10,
                                  restore_best_weights=True, verbose=1)

    l2_param = l2(1e-4)

    def __init__(self, data_wrapper, campaign_id):
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size,pooling='avg')
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)
        x = self.base_model.model.output

        output= self.classification_model(x)

        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
        self.make_trainable_base_model_last_layers(50, remove_normalization=False)
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

    def classification_model(self, x) :
        return x


class T0(TestClassif):
    record_name = "classif-0.2"
    def classification_model(self, x):
        x = Dropout(0.2)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)


class T1(TestClassif):
    record_name = "classif-1024-0.2-512-0.2"
    def classification_model(self, x):
        x = Dense(1024, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.2)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)

class T2(TestClassif):
    record_name = "classif-512-0.2-256-0.2"
    def classification_model(self, x):
        x = Dense(512, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.2)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)

class T3(TestClassif):
    record_name = "classif-1024-0.4"
    def classification_model(self, x):
        x = Dense(1024, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.4)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)

class T4(TestClassif):
    record_name = "classif-1024-0.2"
    def classification_model(self, x):
        x = Dense(1024, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.2)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)


class T5(TestClassif):
    record_name = "classif-512-0.4"
    def classification_model(self, x):
        x = Dense(512, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.4)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)




class T6(TestClassif):
    record_name = "classif-512-0.2"
    def classification_model(self, x):
        x = Dense(512, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.2)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)


class T7(TestClassif):
    record_name = "classif-256-0.4"
    def classification_model(self, x):
        BatchNormalizationV1
        x = Dense(256, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.4)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)

class T8(TestClassif):
    record_name = "classif-256-0.2"
    def classification_model(self, x):
        x = Dense(256, activation='relu',kernel_regularizer=self.l2_param)(x)
        x = Dropout(0.2)(x)
        return Dense(12, activation='softmax',kernel_regularizer=self.l2_param, name='main')(x)



#### Campaign init
models = [
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8
  ]

campaign= lm.campaign.Campaign(campaign_id='test-classifpatient', data_wrapper=data_wrapper, models=models)

#### train
#### models will be serialized in RECORD_DIR (launch only once by campaign)

#campaign.train_all()


#### evaluate
#### models will be loaded and results saved in FIGURE_DIR/campaign_id

campaign.evaluate_and_build_reports()