
import src.features as lf
import src.models as lm
from src.models.stages2 import *


#### Directories
lm.models.RECORD_DIR='../models/records'
lm.models.FIGURE_DIR='../reports/figures'

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')

#### Campaign init
models = [
  stage1.CNN,
  # stage1.MobileNetv3,
  # stage1.ResNetv2,
  # stage2.CNN,
  # stage2.MobileNetv3,
  # stage2.ResNetv2,
  # stage3.MobileNetv3,
  # stage3.ResNetv2,
  # stage4.CNN,
  # stage4.MobileNetv3,
  # stage4.ResNetv2
  ]
campaign= lm.campaign.Campaign(campaign_id='test', data_wrapper=data_wrapper, models=models)

#### train
#### models will be serialized in RECORD_DIR (launch only once by campaign)

campaign.train_all()


#### evaluate
#### models will be loaded and results saved in FIGURE_DIR/campaign_id

#campaign.evaluate_and_build_reports()