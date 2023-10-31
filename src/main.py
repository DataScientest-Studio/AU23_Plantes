
import src.features as lf
import src.models as lm
import src.visualization as lv
from src.models.stages import *
lm.models.RECORD_DIR='../models/records'

#### Data building
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')

#### Train Campaigns
campaign_id='test'

#### Stage 1
#stage1_mobilenet = stage1.Stage1MobileNetv3(data_wrapper)
#stage1_mobilenet.fit_or_load(campaign_id=campaign_id, training=False)
#stage1_resnet = stage1.Stage1ResNetv2(data_wrapper)
#stage1_resnet.fit_or_load(campaign_id=campaign_id, training=True)

#### Stage2

#stage2_mobilenet = stage2.Stage2MobileNetv3(data_wrapper)
#stage2_mobilenet.fit_or_load(campaign_id=campaign_id, training=False)
#stage2_resnet = stage2.Stage2ResNetv2(data_wrapper)
#stage2_resnet.fit_or_load(campaign_id=campaign_id, training=True)

#### Stage3

#stage3_mobilenet = stage3.Stage3MobileNetv3(data_wrapper)
#stage3_mobilenet.fit_or_load(campaign_id=campaign_id, training=False)
#stage3_resnet = stage3.Stage3ResNetv2(data_wrapper)
#stage3_resnet.fit_or_load(campaign_id=campaign_id, training=True)



stage4_mobilenet = stage4.Stage4MobileNetv3(data_wrapper)
stage4_mobilenet.fit_or_load(campaign_id=campaign_id, training=True)


#stage1_mobilenet.evaluate()
#stage1_mobilenet.print_classification_report()
#stage1_mobilenet.display_history_graphs()
#stage1_mobilenet.display_confusion_matrix()
#stage1_mobilenet.display_samples(nb=3)
#stage1_mobilenet.display_samples(nb=6, gradcam=True)


