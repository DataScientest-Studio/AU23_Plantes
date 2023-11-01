
import src.features as lf
import src.models as lm
import src.visualization as lv
from src.models.stages import *
import tensorflow as tf
lm.models.RECORD_DIR='../models/records'
lm.models.FIGURE_DIR='../reports/figures'

data = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')

campaign_id='test'
models = [
  stage1.Stage1MobileNetv3,
  stage1.Stage1ResNetv2,
  stage2.Stage2MobileNetv3,
  stage2.Stage2ResNetv2,
  stage3.Stage3MobileNetv3,
  stage3.Stage3ResNetv2,
  stage4.Stage4MobileNetv3,
  stage4.Stage4ResNetv2
  ]

active_models = []

def train_all() :
    """
    train all the models in the list
    do not keep reference on them to free memory between each
    :return:
    """
    for model in models:
        m= model(data, campaign_id)
        m.fit_or_load(training=True)
        tf.keras.backend.clear_session()


def evaluate_and_build_reports() :
    for model in models:
        m= model(data, campaign_id)
        m.fit_or_load(training=False)
        m.save_evaluation_reports()
        active_models.append(m)
    lv.graphs.compare_models_confusions(active_models, save=True, fig_dir=lm.models.FIGURE_DIR)
    lv.graphs.compare_models_performances(active_models, save=True, fig_dir=lm.models.FIGURE_DIR)







