import sys
sys.path.append("/Users/morph/Desktop/Projet_DS/AU23_Plantes")

import src.features as lf
import src.models as lm
import src.visualization as lv
from src.models.models import *
lm.models.RECORD_DIR='../models/records'

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('/Users/morph/Desktop/Projet_DS/dataset')

#### Train Campaigns
campaign_id='test'

#### Stage 1
stage1_mobilenet = Stage1MobileNetv3(data_wrapper)
stage1_mobilenet.fit_or_load(campaign_id=campaign_id, training=False)
stage1_mobilenet.evaluate()
stage1_mobilenet.print_classification_report()
stage1_mobilenet.display_history_graphs()
stage1_mobilenet.display_confusion_matrix()
stage1_mobilenet.display_samples(nb=3)
stage1_mobilenet.display_samples(nb=6, gradcam=True)


