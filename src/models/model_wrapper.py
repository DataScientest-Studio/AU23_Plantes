import tensorflow as tf

from tensorflow import keras
from keras import Model, Input
from keras.regularizers import l2

class BaseModelWrapper:
    """
    A BaseModelWrapper encapsulates a  pretrained base model  and its main characteristics
    All BaseModelWraper object must assign the variables :
    preprocessing, log_name, model, grad_cam_layer
    """
    preprocessing = None
    log_name: str = None
    model: Model = None
    grad_cam_layer: str = None

    def __init__(self):
        pass


class VoidBaseModel(BaseModelWrapper):
    def __init__(self, img_size: tuple) -> None:
        self.preprocessing = lambda x: x
        self.log_name = 'none'
        inputs = Input(shape=(None,))
        self.model = Model(inputs=inputs, outputs=inputs)
        self.grad_cam_layer = None

class SimpleCNN(BaseModelWrapper):
    def __init__(self, img_size: tuple) -> None:
        self.preprocessing = lambda x: x / 255.
        self.log_name = 'CNN'

        inputs = tf.keras.layers.Input(shape=img_size + (3,))
        x = inputs

        x = tf.keras.layers.Conv2D(32, (6, 6), activation='relu',
                                   padding='same', kernel_regularizer=l2(0.001),
                                   input_shape=(224,224,3)
                                   )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu',
                                   padding='same', kernel_regularizer=l2(0.001)
                                   )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(128, (4, 4), activation='relu',
                                   padding='same', kernel_regularizer=l2(0.001)
                                   )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                   padding='same', kernel_regularizer=l2(0.001), name='Conv_1'
                                   )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

        x = tf.keras.layers.Flatten()(x)

        outputs = x

        self.model = Model(inputs=inputs, outputs=outputs)

        self.grad_cam_layer = 'Conv_1'


class MobileNetv2(BaseModelWrapper):
    def __init__(self, img_size: tuple, pooling:str=None) -> None:
        self.preprocessing = tf.keras.applications.mobilenet_v2.preprocess_input
        self.log_name = 'mobilenetv2'
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=img_size + (3,),
            include_top=False,
            alpha=1.0,
            weights='imagenet',
            pooling=pooling
        )
        self.grad_cam_layer = 'Conv_1'


class MobileNetv3(BaseModelWrapper):
    def __init__(self, img_size: tuple,pooling:str=None) -> None:
        self.preprocessing = tf.keras.applications.mobilenet_v3.preprocess_input
        self.log_name = 'mobilenetv3'
        self.model = tf.keras.applications.MobileNetV3Large(
            input_shape=img_size + (3,),
            include_top=False,
            alpha=1.0,
            weights='imagenet',
            pooling=pooling
        )
        self.grad_cam_layer = 'Conv_1'


class ResNet50V2(BaseModelWrapper):
    def __init__(self, img_size: tuple,pooling:str=None) -> None:
        self.preprocessing = tf.keras.applications.resnet50.preprocess_input
        self.log_name = 'resnetv2'
        self.model = tf.keras.applications.ResNet50(
            input_shape=img_size + (3,),
            include_top=False,
            weights='imagenet',
            pooling=pooling
        )
        self.grad_cam_layer = 'Conv_1'


def get_gradcam_encapsulation(model_wrapper: BaseModelWrapper) -> Model:
    """
    Useful for GradCam computation while doing fine-tuning with the training=false argument

    Returns a model encapuslating the base model defined in model_wrapper
    and having 2 ouputs :
        - first output is the base model output
        - second output is the output of the last convolutional layer as described in the model_wrapper
    """
    base_model = model_wrapper.model
    return Model(inputs=base_model.input,
                 outputs=[base_model.output,
                          base_model.get_layer(model_wrapper.grad_cam_layer).output],
                 name='gradcam')
