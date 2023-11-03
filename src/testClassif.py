
import src.features as lf
import src.models as lm
from src.models.stages import *
import src.testClassifModels as tests

#### Directories
lm.models.RECORD_DIR='../models/records'
lm.models.FIGURE_DIR='../reports/figures'

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')

#### Campaign init
models = [
  tests.T1,
  # tests.T2,
  # tests.T3,
  # tests.T4,
  # tests.T5,
  # tests.T6,
  # tests.T7,
  ]
campaign= lm.campaign.Campaign(campaign_id='test-classif2', data_wrapper=data_wrapper, models=models)

#### train
#### models will be serialized in RECORD_DIR (launch only once by campaign)

campaign.train_all()


#### evaluate
#### models will be loaded and results saved in FIGURE_DIR/campaign_id

#campaign.evaluate_and_build_reports()