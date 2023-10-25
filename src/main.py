
import src.features as lf
import src.models as lm
import src.visualization as lv
from src.models.models import *




# construction des données
data_wrapper = lf.data_builder.create_dataset_from_directory('../data/v2-plant-seedlings-dataset/')

step1mobilenet = Step1MobileNetv3(data_wrapper)
## TODO debug
step1mobilenet.train()
print (type(step1mobilenet))


# step1mobilenet.train()
# step1mobilenet.serialize('test')
# step1mobilenet.evaluate()




# entrainement des modèles

