import tensorflow as tf

from tensorflow import keras
from keras import Model, Input


class BaseModelWrapper :
    """
    A BaseModelWrapper encapsulates a  pretrained base model  and its main characteristics
    All BaseModelWraper object must assign the variables :
    preprocessing, log_name, model, grad_cam_layer
    """
    preprocessing = None
    log_name:str= None
    model:Model = None
    grad_cam_layer:str = None

    def __init__(self):
        pass


class VoidBaseModel(BaseModelWrapper) :
    def __init__(self, img_size:tuple) -> None:
        self.preprocessing = lambda x : x
        self.log_name = 'none'
        inputs = Input(shape=(None,))
        self.model = Model(inputs=inputs, outputs=inputs)
        self.grad_cam_layer = None


from keras.applications import mobilenet_v2
from keras.applications import MobileNetV2
class MobileNetv2(BaseModelWrapper) :
    def __init__(self, img_size:tuple) -> None:
        self.preprocessing = mobilenet_v2.preprocess_input
        self.log_name = 'mobilenetv2'
        self.model = MobileNetV2(
                input_shape=img_size + (3,),
                include_top=False,
                alpha=1.0,
                weights='imagenet',
                pooling='avg'
            )
        self.grad_cam_layer = 'Conv_1'




from keras.applications import mobilenet_v3
from keras.applications import MobileNetV3Large
class MobileNetv3(BaseModelWrapper):
    def __init__(self, img_size:tuple) -> None:
        self.preprocessing = mobilenet_v3.preprocess_input
        self.log_name = 'mobilenetv3'
        self.model = MobileNetV3Large(
                input_shape=img_size + (3,),
                include_top=False,
                alpha=1.0,
                weights='imagenet',
                pooling='avg'
        )
        self.grad_cam_layer = 'Conv_1'






class ResNet50V2(BaseModelWrapper):
    def __init__(self, img_size:tuple) -> None:
        self.preprocessing = tf.keras.applications.resnet50.preprocess_input
        self.log_name = 'resnetv2'
        self.model = tf.keras.applications.ResNet50(
                input_shape=img_size + (3,),
                include_top=False,
                weights='imagenet',
                pooling='avg'
        )
        self.grad_cam_layer = 'Conv_1'






def get_gradcam_encapsulation(model_wrapper:BaseModelWrapper) -> Model:
    """
    Useful for GradCam computation while doing fine-tuning with the training=false argument

    Returns a model encapuslating the base model defined in model_wrapper
    and having 2 ouputs :
        - first output is the base model output
        - second output is the output of the last convolutional layer as described in the model_wrapper
    """
    base_model = model_wrapper.model
    return Model(inputs=base_model.input,
                 outputs = [base_model.output,
                            base_model.get_layer(model_wrapper.grad_cam_layer).output],
                 name='gradcam')