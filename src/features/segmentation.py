
from skimage.morphology import closing
from skimage.morphology import disk 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


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
    def __init__(self,) -> None   :
        self.images     :tuple[np.ndarray, np.ndarray] = None
        self.threshold  :list[int, int] = None
        self.radius     :float  = None
        self.method     :str    = None
        self.color      :str    = None

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
                    value       : list[float, float, float] = [1., 1., 1.],
                    mask        : bool = False
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
        self.mask        = self.getMask()
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
            self.new_img = self.BackgroundColor(img=self.new_img, lower_color=lower_color, 
                                              upper_color=upper_color, value=value)
        elif self.color == 'black':
            # if color is set on <black>
            self.new_img = self.img_seg 
        else:  
            # if image not in [white, black]
            self.new_img = None 

        if not mask :   
            return self.new_img
        else: 
            return self.new_img, self.mask
class FinalProcess(ImageProcess):
    def __init__(self, 
                threshold   : list[int, int]  = [0, 80], # list of values extracted from histogram colors 
                radius      : float = 2.,                # a float number used for dilation and erosion operation
                method      : str   = 'numpy',           # method used for segmenation: method can be : <where> or <numpy>
                color       : str   = 'white'            # color is string type: takes also two values : <white> or <black>
                ) -> None   :
        super().__init__()
        self.threshold  = threshold
        self.radius     = radius
        self.method     = method
        self.color      = color
        
    def imgSegmentation(self,
            x           : any,
            upper_color : list[int, int, int]       = [30, 30, 30],  
            lower_color : list[int, int, int]       = [0, 0, 0],
            value       : list[float, float, float] = [1., 1., 1.],
            mask        : bool =False
            ) -> np.ndarray:
        
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
        self.SHAPE = x.shape

        if len(self.SHAPE) > 3:  self.shape = self.SHAPE[1:]
        else: self.shape = self.SHAPE

        # getting image type 
        self.dtype = x.dtype
        # converting image in a numpy array type 
        self.is_numpy_array = False 

        # converting image in a numpy array type 
        if type(x) == type(np.array([])):
            self.is_numpy_array = True 
            self.image  = x.astype('float32')
        else:
            self.image  = x.numpy().astype('float32')
        
        self.image = self.image.reshape(self.shape)
        # converting image from RGB to RGB-LAB
        self.image_lab      = RGB2LAB_SPACE(image=self.image.copy())
        # creating a tuple for the next process
        self.images         = (self.image, self.image_lab)
        # running process 
        if not mask:
            self.image          = self.Segmentation(upper_color, lower_color, value, mask=mask)
        else:
            self.image, self.Mask = self.Segmentation(upper_color, lower_color, value, mask=mask)
        try:
            # converting image format from numpy array type to tf.tensor 
            if self.is_numpy_array is False:
                self.x = tf.constant(self.image, dtype=self.dtype, shape=self.SHAPE)
            else:
                self.x = self.image.astype(self.dtype).reshape(self.SHAPE)
            if not mask:
                # returning return 
                return self.x
            else:
                return self.x, self.Mask
        except TypeError:
            raise ValueError("None type cannot be converted in tensorflow tensor.\
                    Please check the color or lower and upper range values and try again")

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

def remove_background(
        x           : any, 
        color       : str   = 'white', 
        radius      : float = 4.,
        threshold   : list[int, int] = [0, 80],
        mask        : bool = False
        ) -> np.ndarray:


    x = FinalProcess(
        threshold   =threshold,
        color       =color, 
        radius      =radius
        ).imgSegmentation(x=x, mask=mask)
    
    return x