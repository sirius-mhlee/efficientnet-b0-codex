import Config

from Model.CustomAlexnet import *
from Model.EfficientNet_B0 import *

def get_model_by_name(model_name):
    if model_name == 'alexnet':
        return CustomAlexnet(class_num=Config.class_num)
    elif model_name == 'efficientnetb0':
        return EfficientNet_B0(class_num=Config.class_num)
