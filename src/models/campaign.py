import src.features as lf
import src.models as lm
import src.visualization as lv
from src.models.stages import *
import tensorflow as tf
import os

class Campaign() :
    """
    Initializes a new campaign with the given data wrapper, campaign ID, and models.

    Parameters:
        data_wrapper (lf.data_builder.ImageDataWrapper): The data wrapper object.
        campaign_id (str): The ID of the campaign.
        models (list[lm.models.Trainer]): A list of trainer model classes to use (not initialized).

    Returns:
        None
    """
    def __init__(self, campaign_id:str, data_wrapper:lf.data_builder.ImageDataWrapper,  models:list[lm.models.Trainer]):
        self.data = data_wrapper
        self.campaign_id = campaign_id
        self.models = models

    active_models = []

    def train_all(self) :
        """
        train all the models in the list
        do not keep reference on them to free memory between each
        :return:
        """
        for model in self.models:
            m= model(self.data, self.campaign_id)
            m.fit_or_load(training=True)
            tf.keras.backend.clear_session()

    def evaluate_and_build_reports(self, gradcam=True) :
        fig_dir = lm.models.FIGURE_DIR
        campaign_dir = f"{fig_dir}/{self.campaign_id}"
        
        if not os.path.exists(campaign_dir):
            os.makedirs(campaign_dir)
        for model in self.models:
            m= model(self.data, self.campaign_id)
            m.fit_or_load(training=False)
            m.save_evaluation_reports(gradcam)
            self.active_models.append(m)
        self.compare_models_confusions(save=True)
        self.compare_models_performances(save=True)

    def compare_models_performances(self,save=False) :
        lv.graphs.compare_models_performances(self.active_models, save=save, fig_dir=lm.models.FIGURE_DIR)

    def compare_models_confusions(self, save= False):
        lv.graphs.compare_models_confusions(self.active_models, save=save, fig_dir=lm.models.FIGURE_DIR)