
import src.features as lf
import src.models as lm
from src.models.stages import *


#### Directories
lm.models.RECORD_DIR='../models/records'
lm.models.FIGURE_DIR='../reports/figures'

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')

#### Campaign init
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
campaign= lm.campaign.Campaign(campaign_id='test', data_wrapper=data_wrapper, models=models)

#### train
#### models will be serialized in RECORD_DIR (launch only once by campaign)

campaign.train_all()


#### evaluate
#### models will be loaded and results saved in FIGURE_DIR/campaign_id

campaign.evaluate_and_build_reports()
