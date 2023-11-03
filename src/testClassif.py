
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







#### Directories
lm.models.RECORD_DIR='../models/records'
lm.models.FIGURE_DIR='../reports/figures'

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')






class TestClassif(lm.models.Trainer):
    # abstract class
    record_name = "none"
    lr2=0.0001
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,  # min_delta=1e-3,
                                     patience=3,  # tf.math.ceil(epoch1/10),
                                     verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=5,  # min_delta=5e-3,#tf.math.ceil(epoch1/5),
                                  restore_best_weights=True, verbose=1)

    def __init__(self, data_wrapper, campaign_id):
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)
        inputs = tf.keras.layers.Input([None, None, 3])
        x= self.base_model.model(inputs, training=False)
        #x = self.base_model.model.output
        x = GlobalAveragePooling2D()(x)
        x= self.classification_model(x)
        output = Dense(12, activation='softmax', name='main')(x)
        #self.model = Model(inputs=self.base_model.model.input, outputs=output)
        self.model = Model(inputs=inputs, outputs=output)

    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

        #self.make_trainable_base_model_last_layers(50)
        self.base_model.model.trainable = True
        self.compile_fit(lr=self.lr2, epochs=self.epoch2, is_fine_tuning=True)

        #self.make_trainable_base_model_last_layers(50)

    def classification_model(self, x) :
        return x


class T1(TestClassif):
    record_name = "classif-512-0.2-256-0.2"
    def classification_model(self, x):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        return x

class T2(TestClassif):
    record_name = "classif-512-0.5-256-0.5"
    def classification_model(self, x):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x


class T3(TestClassif):
    record_name = "classif-256-0.5-128-0.5"
    def classification_model(self, x):
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x


class T4(TestClassif):
    record_name = "classif-1024-0.5-512-0.5"
    def classification_model(self, x):
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x


class T5(TestClassif):
    record_name = "classif-512-0.5"
    def classification_model(self, x):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x

class T6(TestClassif):
    record_name = "classif-256-0.5"
    def classification_model(self, x):
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x


class T7(TestClassif):
    record_name = "classif-128-0.5"
    def classification_model(self, x):
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x







#### Campaign init
models = [
  T1,
  # T2,
  # T3,
  # T4,
  # T5,
  # T6,
  # T7,
  ]
campaign= lm.campaign.Campaign(campaign_id='test-classif2', data_wrapper=data_wrapper, models=models)

#### train
#### models will be serialized in RECORD_DIR (launch only once by campaign)

#campaign.train_all()


#### evaluate
#### models will be loaded and results saved in FIGURE_DIR/campaign_id

#campaign.evaluate_and_build_reports()