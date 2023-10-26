
import src.features as lf
import src.models as lm
import src.visualization as lv
from src.models.models import *




# Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')


# Train Campaigns
campaign_id='test'

step1mobilenet = Step1MobileNetv3(data_wrapper)
step1mobilenet.fit_or_load(campain_id=campaign_id, training=False)
step1mobilenet.evaluate()

