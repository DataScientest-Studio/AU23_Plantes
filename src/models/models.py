from sklearn.metrics import classification_report

import src.models as lm
import src.visualization as lv
import src.features as lf

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras import Model, Input, Sequential
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, Hinge
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import DataFrameIterator
from keras.utils import img_to_array, load_img

import pickle

""" Model training """

RECORD_DIR: str = "../models/records"

class Trainer():
    #abstract class
    """
    Trainers are classes making model init, training and estimation
    """
    # default parameters


    img_size: tuple = (224, 224)
    data_augmentation = dict(
        validation_split=0.12,
        rotation_range=360,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
    )
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                     patience=2, verbose=1),
    stop_callback = EarlyStopping(monitor='val_loss', patience=3,
                                  restore_best_weights=True, verbose=1)
    epoch1: int = 1
    lr1: float = 0.001
    # used for 2 rounds fine tuning
    epoch2: int = 1
    lr2: float = 0.00001

    batch_size = 32

    # other attributes that will be init by subclasses
    record_name: str = None
    base_model: lm.model_wrapper.BaseModelWrapper = None
    model: Model = None
    data: lf.data_builder.ImageDataWrapper = None
    train: DataFrameIterator = None
    validation: DataFrameIterator = None
    test: DataFrameIterator = None
    history1: dict = None
    history2: dict = None
    results: pd.DataFrame = None



    def __init__(self, data_wrapper):
        self.data = data_wrapper
        # build data flows
        self.train, self.validation, self.test = lf.data_builder.get_data_flows(self.data, self.base_model,
                                                                                self.batch_size, self.data_augmentation,
                                                                                self.img_size)

    def serialize(self, campain_id=None):
        """
        Serializes the model and history objects to disk.

        Parameters:

        timestamp_str (str): Optional timestamp or ID string used to create the directory for storing the serialized files.
        If not provided, the files will be stored in the main record directory.

        Returns:
            List[str]: A list of file paths containing the serialized model and history filenames.
        """
        history1_file, history2_file, model_file = self.get_filenames(campain_id)
        self.model.save(model_file)
        with open(history1_file, 'wb') as f:
            pickle.dump(self.history1, f)
        if (self.history2):
            with open(history2_file, 'wb') as f:
                pickle.dump(self.history2, f)
        return [model_file, history1_file, history2_file]

    def get_filenames(self, campain_id):
        if (campain_id):
            path = f"{RECORD_DIR}/{campain_id}/{self.record_name}"
        else:
            path = f"{RECORD_DIR}/{self.record_name}"
        model_file = path + '_model.h5'
        history1_file = path + '_history1.pkl'
        history2_file = path + '_history2.pkl'
        return history1_file, history2_file, model_file

    def deserialize(self, campaign_id):
        """
        Deserialize the model, history1, and history2 from the given timestamp string.

        :param campaign_id: The timestamp string representing the desired snapshot.
        :type campaign_id: str

        :return: None
        """
        history1_file, history2_file, model_file = self.get_filenames(campaign_id)
        self.model = keras.models.load_model(model_file)
        with open(history1_file, 'rb') as f:
            self.history1 = pickle.load(f)
        if (self.history2):
            with open(history2_file, 'rb') as f:
                self.history2 = pickle.load(f)

    def print_step(self, step_name: str) -> str:
        print(f">>> {self.record_name} –– {step_name}")


    def fit_or_load(self, campain_id=None, training=True):
        """
        Fit or load the model for training or inference.
        Parameters:
            campain_id (optional): The ID of the campaign to load or serialize.
            training (bool): A flag indicating whether to perform training or inference.
        Returns:
            None
        """
        if (training):
            self.print_step("Training")
            self.process_training()
            self.print_step("Serialize ")
            self.serialize(campain_id=campain_id)
        else:
            self.print_step("Loading")
            self.deserialize(campaign_id=campain_id)

    def process_training(self):
        """
        implement the training sequence using compile_fit method
        must be implemented by subclasses
        """
        raise NotImplementedError("process_training must be implemented")   

    def compile_fit(self, lr: float, epochs: int):
        """
                Train the model and keep trace of history
                The model can then be serialized
                Parameters:
                    epochs (int): The number of epochs to train the model for
                Returns:
                    None
        """
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=CategoricalCrossentropy(),
            metrics=CategoricalAccuracy()
        )

        self.history1 = self.model.fit(
            self.train,
            validation_data=self.validation,
            epochs=epochs,
            batch_size=self.batch_size,
            class_weight=self.data.weights,
            callbacks=[self.lr_reduction, self.stop_callback]
        ).history


    def single_prediction(self, path: str) -> str:
        """
        Evaluate the model on a single image.
        Parameters:
            path (str): The path to the image.
        Returns:
            str: The predicted class
        """
        self.print_step("Evaluation")
        img = load_img(path, target_size=self.img_size)
        img = img_to_array(img)
        img = self.base_model.preprocessing(img)
        pred_vector = self.model.predict(np.expand_dims(img, axis=0))
        return self.data.classes[np.argmax(pred_vector)]


    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate the model by getting predictions and storing them in the results attribute.
        Parameters:
            None (it uses the test flow encapsultated by the Trainer
        Returns:
            pd.DataFrame: The results dataframe
        """
        self.print_step("Evaluation")
        self.results = lf.data_builder.get_predictions_dataframe(self.model, self.test, self.data.test_df)
        return self.results


    def print_classification_report(self) -> str:
        """
        Print the classification report and return it as a string.
        """
        self.print_step("Classification Report")
        cr = classification_report(self.results.actual, self.results.predicted)
        print(cr)
        return cr

    def display_history_graphs(self) -> None:
        """
        Display the graphs of the training and validation history.
        """
        self.print_step("Display training history graphs")
        lv.graphs.plot_history_graph(self.history1, self.history2, self.record_name)

    def display_samples(self, nb: int, include_true_pred = True, include_false_pred = True, gradcam: bool = False) -> None :
        """
        Display training data samples with gradcam if gradcam is True.

        Parameters:
            nb (int): The number of samples to display.
            include_true_pred (bool, optional): Whether to include samples with true predictions. Defaults to True.
            include_false_pred (bool, optional): Whether to include samples with false predictions. Defaults to True.
            gradcam (bool, optional): Whether to show GradCAM layer or not. Defaults to False.

        Returns:
            None
        """
        self.print_step("Display training data samples")
        results = self.results
        if (include_true_pred and not include_false_pred): results = results[results['Same']==True]
        if (not include_true_pred and  include_false_pred): results = results[results['Same']==False]
        if gradcam :
            lv.graphs.display_results(results, nb=nb, record_name=self.record_name, gradcam=True, model=self.model,
                                      base_model_wrapper=self.base_model, img_size=self.img_size)
        else : lv.graphs.display_results(results, nb=nb, record_name=self.record_name)

    def display_confusion_matrix(self) -> None:
        """
       This function display the confusion matrix for the results of the model.
       It uses the `plot_confusion_matrix` function from the `lv.graphs` module to plot the matrix.

        Parameters:
        - None

        Returns:
        - None
        """
        self.print_step("Display confusion matrix")
        lv.graphs.plot_confusion_matrix(self.results, self.record_name)






""""


    
    Stage 1 : simple training (no fine tuning, no background removal, simple classification)
    comparison between the 2 selected models : Mobilnetv3 and Resnet




"""

class Stage1(Trainer):
    # abstract class
    record_name = "none"

    def __init__(self, data_wrapper):
        super().__init__(data_wrapper)
        x = self.base_model.model.output
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
            self.base_model.model.trainable = False
            self.compile_fit(lr=self.lr1, epochs=self.epoch1)



class Stage1MobileNetv3(Stage1):
     record_name = "Stage-1_MobileNetv3"

     def __init__(self, data_wrapper):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        super().__init__(data_wrapper)


class Stage1ResNetv2(Stage1):
    record_name = "Stage-1_ResNetv2"

    def __init__(self, data_wrapper):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size)
        super().__init__(data_wrapper)


"""

    Stage 2 : simple training (no fine tuning, simple classification)
    with background removal
    comparison between the 2 selected models : Mobilnetv3 and Resnet


"""

class Stage2(Trainer):
    # abstract class
    record_name = "none"

    def __init__(self, data_wrapper):
        super().__init__(data_wrapper)
        x = self.base_model.model.output
        x = Dropout(0.2)(x)
        output = Dense(12, activation='softmax', name='main')(x)
        self.model = Model(inputs=self.base_model.model.input, outputs=output)

    def process_training(self):
        self.base_model.model.trainable = False
        self.compile_fit(lr=self.lr1, epochs=self.epoch1)

class Stage2MobileNetv3(Stage1):
    record_name = "Stage-2_MobileNetv3"

    def __init__(self, data_wrapper):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.MobileNetv3(self.img_size)
        def preprocessing(x):
            return self.base_model.preprocessing(remove_background(x))

        self.base_model.preprocessing = preprocessing
        super().__init__(data_wrapper)

class Stage2ResNetv2(Stage1):
    record_name = "Stage-2_ResNetv2"

    def __init__(self, data_wrapper):
        # set the base model -- must be set before super().__init__()
        self.base_model = lm.model_wrapper.ResNet50V2(self.img_size)
        def preprocessing(x):
            return self.base_model.preprocessing(remove_background(x))

        self.base_model.preprocessing= preprocessing
        super().__init__(data_wrapper)

def RGB2LAB_SPACE(image : np.ndarray):

    """
    args:
        image : image.shape = (m, m, 3)
    return:
        filter : filter.shape = image.shape 
    """

    import cv2

    filter  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    filter[:, :, 0] = cv2.normalize(filter[:, :, 0], None, 0, 1, cv2.NORM_MINMAX)
    filter[:, :, 1] = cv2.normalize(filter[:, :, 1], None, 0, 1, cv2.NORM_MINMAX)
    filter[:, :, 2] = cv2.normalize(filter[:, :, 2], None, 0, 1, cv2.NORM_MINMAX)

    return filter

class ImageProcess:
    def __init__(self, 
                images      : tuple[np.ndarray, np.ndarray], 
                threshold   : list[int, int]  = [0, 80], 
                radius      : float = 2.,
                method      : str   = 'numpy',
                color       : str   = 'white'
                ) -> None   :
        
        self.images     = images
        self.threshold  = threshold
        self.radius     = radius
        self.method     = method
        self.color      = color
        
    def getMask(self,) -> np.ndarray:

        """
        arg:
            None
        return:
            filter : np.ndarray
        """
        
        from skimage.morphology import closing
        from skimage.morphology import disk  

        # numpy method
        if self.method == "numpy": 
            self.filter = ( self.images[1][..., 1] > self.threshold[0] / 255. ) &\
                    ( self.images[1][..., 1] < self.threshold[1] / 255. ) 
        # where method
        else: 
            self.filter = np.where( ( self.images[1][..., 1] > self.threshold[0] /255.  ) &\
                    ( self.images[1][..., 1] < self.threshold[1] /255. ), 1, 0)

        # create a disk
        self.DISK = disk(self.radius)
   
        # create filter by comparing the values close to DISK or inside the disk
        self.filter = closing(self.filter, self.DISK)
         
        # returning values
        return self.filter
    
    def BackgroundColor(self, 
                img : np.ndarray, 
                upper_color : list[int, int, int], 
                lower_color : list[int, int, int],
                value       : list[float, float, float] = [1., 1., 1.]
                ) -> np.ndarray :
        
        """
        arg:
            (img, upper_color, lower_color, value)
            * img is np.ndarray type of (m, m, c) here c is the channel and m is the image size
            * upper_color is a list of value used t0 fix the maximum along (r, g, b)
            * lower_color is a list of value used to fix the minimum along (r, g, b)
            * values is a list of normalized pixels used to change the background 

        return:
            None or np.ndarray
        """

        import cv2

        # image shape
        shape       = img.shape
        #Normalizing lower and upper color and reshaping them in shape[-1]
        # this means that if img.shape = (m, m, 3) -->  upper_color.shape = (3, ) 
        # we have to do it because we have 3 channels in this case 
        upper_color = np.array(upper_color).reshape((shape[-1], )) / 255.
        lower_color = np.array(lower_color).reshape((shape[-1], )) / 255.

        try:
            mask = cv2.inRange(src=img, lowerb=lower_color, upperb=upper_color)
            img[mask > 0] = value

            # returning the value 
            return  img
        except TypeError: return None 

    def Segmentation(self,
                    upper_color : list[int, int, int], 
                    lower_color : list[int, int, int],
                    value       : list[float, float, float] = [1., 1., 1.]
                    ) -> np.ndarray :
        
        """
        arg:
            (upper_color, lower_color, value)
            * upper_color is a list of value used t0 fix the maximum along (r, g, b)
            * lower_color is a list of value used to fix the minimum along (r, g, b)
            * values is a list of normalized pixels used to change the background 

        return:
            None or np.ndarray
        """

        # original image 
        self.img_seg     = self.images[0]
        # original image shape 
        self.shape       = self.images[0].shape

        # creating mask from image_lab
        self.mask        = ImageProcess(self.images, self.threshold, self.radius, self.method, self.color).getMask()
        # converting mask in a float type
        self.mask        = self.mask * 1.

        #################################################
        ########### segmentation section  ###############
        #################################################
        # applying mask of the orginal image
        self.img_seg[..., 0] = self.img_seg[..., 0] * self.mask * 1.
        self.img_seg[..., 1] = self.img_seg[..., 1] * self.mask * 1.
        self.img_seg[..., 2] = self.img_seg[..., 2] * self.mask * 1.

        # change background color 
        if self.color == 'white':
            # if color is set on <white>
            self.new_img = self.img_seg .reshape(self.shape[0], self.shape[1], 3)
            self.new_img = ImageProcess(
                    self.images, self.threshold, self.radius, self.method, self.color
                            ).BackgroundColor(img=self.new_img, lower_color=lower_color, 
                                              upper_color=upper_color, value=value)
        elif self.color == 'black':
            # if color is set on <black>
            self.new_img = self.img_seg 
        else:  
            # if image not in [white, black]
            self.new_img = None 
            
        return self.new_img
    
class FinalProcess:
    def __init__(self, 
                threshold   : list[int, int]  = [0, 80], # list of values extracted from histogram colors 
                radius      : float = 2.,                # a float number used for dilation and erosion operation
                method      : str   = 'numpy',           # method used for segmenation: method can be : <where> or <numpy>
                color       : str   = 'white'            # color is string type: takes also two values : <white> or <black>
                ) -> None   :
        
        self.threshold  = threshold
        self.radius     = radius
        self.method     = method
        self.color      = color
        
    def imgSegmentation(self,
            x           : any,
            upper_color : list[int, int, int]       = [30, 30, 30],  
            lower_color : list[int, int, int]       = [0, 0, 0],
            value       : list[float, float, float] = [1., 1., 1.]):
        
        """
        arg:
            (x, upper_color, lower_color, value)

            * x is tf.tensor of shape (m, m, c) here c is the channel and m is the image size
            * upper_color is a list of value used t0 fix the maximum along (r, g, b)
            * lower_color is a list of value used to fix the minimum along (r, g, b)
            * values is a list of normalized pixels used to change the background 

        return:
            tf.tensor or None type 
        """

        import tensorflow as tf 

        #####################################################################
        ###############  Semantic image segmentation  section ###############
        #####################################################################

        # getting image dimension 
        self.shape = x.shape
        # getting image type 
        self.dtype = x.dtype

        # converting image in a numpy array type 
        self.image          = x.numpy().astype('float32')
        # converting image from RGB to RGB-LAB
        self.image_lab      = RGB2LAB_SPACE(image=self.image.copy())
        # creating a tuple for the next process
        self.images         = (self.image, self.image_lab)
        # running process 
        self.image          = ImageProcess(self.images, self.threshold, 
                                    self.radius, self.method, self.color).\
                                        Segmentation(upper_color, lower_color, value)

        try:
            # converting image format from numpy array type to tf.tensor 
            self.x = tf.constant(self.image, dtype=self.dtype, shape=self.shape)

            # returning return 
            return self.x
        except TypeError:
            raise ValueError("None type cannot be converted in tensorflow tensor.\
                    Please check the color or lower and upper range values and try again")

def remove_background(
        x           : any, 
        color       : str='white', 
        radius      : float=4.,
        threshold   : list[int, int]=[0, 80]
        ) -> any:
    
    x = FinalProcess(
        threshold   =threshold,
        color       =color, 
        radius      =radius
        ).imgSegmentation(x=x)
    
    return x

def Plot_Histograms(
    data            : pd.DataFrame, 
    figsize         : tuple  = (15, 4),  
    mul             : float  = 1.0,
    select_index    : list   = [0],
    ylabel          : str    = "Intensity (Px)",
    bins            : int    = 20,
    rwidth          : float  = 0.2,
    share_x         : bool   = True,
    share_y         : bool   = False
    ):

    """
    * ----------------------------------------------------------

    arg:
        * data is a dataframe with the path of all images
        * figsize is a tuple used to create figures 
        * color_indexes is a list of size 3 used to set color in each plot
        * mul is numeric value
        * names is a list that contains the names of speces len(names) = n 
        * select_index is a list of values 
        * bins is an integer  
        * rwidth is the size of bins 
    return:
        None

    * ----------------------------------------------------------
    
    >>> filter_selection(data=data, fisize = (8, 8))
    
    """
    import matplotlib.pyplot as plt 

    img     = [RGB2LAB_SPACE(image=plt.imread(data.dataframe.iloc[m].path).astype("float32")) for m in select_index]
    names   = [data.dataframe.label[m] for m in select_index]

    # canaux 
    canaux = ["Luninosity", "Luninosity", "Luninosity"]
    error = None
    # uploading all python colors
    colors = ['darkred', "darkgreen", "darkblue"]
  
    # plotting image in function of the channel
    lenght = len(select_index)
    
    if   lenght > 1  : fig, axes = plt.subplots(lenght, 3, figsize=figsize, sharey=share_y, sharex=share_x)
    elif lenght == 1 : fig, axes = plt.subplots(lenght, 3, figsize=figsize, sharey=share_y, sharex=share_x) 
    else: error = True  

    if error is None:
        if lenght > 1:
            for i in range(lenght): 
                index =  i
                channel = img[index].shape[-1]

                for j in range(channel):
                    axes[i, j].hist(img[index][:, :, j].ravel() * mul, bins=bins, color=colors[j], histtype="bar", 
                                    rwidth=rwidth ,density=False)
                    # title of image
                    if i == 0: axes[i, j].set_title(f"Channel {j}", fontsize="small", weight="bold", color=colors[j])
                    # set xlabel
                    if i == lenght-1 :axes[i, j].set_xlabel(canaux[j], weight="bold", fontsize='small', color=colors[j])
                    # set ylabel
                    axes[i, j].set_ylabel(ylabel, weight="bold", fontsize='small', color=colors[j])
                    # set lelend 
                    axes[i, j].legend(labels = [names[i]], fontsize='small', loc="best")
                    axes[i, j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                else: pass
            else: pass
        else:
            for i in select_index:
                channel = img[i].shape[-1]
                for j in range(channel):
                    axes[j].hist( img[i][:, :, j].ravel(), bins=bins, color=colors[j], histtype="bar", 
                                    rwidth=rwidth ,density=False)

                    # set ylabel
                    axes[j].set_ylabel(ylabel, weight="bold", fontsize='small', color=colors[j])
                    # set title 
                    axes[j].set_title(f"Channel {j}", fontsize="small", weight="bold", color=colors[j])
                    # set xlabel 
                    axes[j].set_xlabel(canaux[j], weight="bold",fontsize='small',color=colors[j])
                    # set legend 
                    axes[j].legend(labels = [names[i]], fontsize='small', loc="best")
                    axes[i, j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.show()
    else: pass

