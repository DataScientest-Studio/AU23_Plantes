import src.models as lm
import src.features as lf
import tensorflow as tf
from tensorflow import keras

from src.models.final_test.stage1 import Stage1
from keras.layers import Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras import Model, Input, Sequential


"""

    Stage 2 : simple training (no fine tuning, simple classification)
    with background removal
    comparison between the 2 selected models : Mobilnetv3 and Resnet


"""

class Stage2(Stage1):

    def __init__(self, data_wrapper, campaign_id):
        self.add_background_removal()
        super().__init__(data_wrapper, campaign_id)




class MobileNetv3(Stage2):
    record_name = "2-noBg-Mob"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size,pooling='avg')
        super().__init__(data_wrapper, campaign_id)


class ResNetv2(Stage2):
    record_name = "2-noBg-Res"
    def __init__(self, data_wrapper, campaign_id):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size,pooling='avg')
        super().__init__(data_wrapper, campaign_id)
