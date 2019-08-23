import os
import sys
import json
import math
from utils import cross_from_points, get_true_angle, three_normalized_dimensions, image_preprocess

script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(script_dir, '..', 'scripts')))


from config import BOXCARS_LIST_TRAIN, BOXCARS_LIST_VAL, BOXCARS_LIST_TEST

image_dir = '/home/vivacityserver6/datasets/BoxCars116k/ims_val'
anns_dir = '/home/vivacityserver6/datasets/BoxCars116k/anns_val'
separator = ' '

def wirte_list_image_path(mode):
    if mode == "train":
        list_image_file = BOXCARS_LIST_TRAIN
    elif mode == "validation":
        list_image_file = BOXCARS_LIST_VAL
    elif mode == "test":
        list_image_file = BOXCARS_LIST_TEST
    else:
        raise Exception('Setting a wrong data mode, should be either training. validation or test')

    file = open(list_image_file, 'w+')

    for folder in os.listdir(image_dir):
        for images in os.listdir(os.path.join(image_dir, folder)):
            label = get_label_from_image_name(os.path.join(folder, images))
            file.write(os.path.join(folder, images) + separator + label + '\n')

    file.close()

def get_label_from_image_name(image_name):
    txt_name = os.path.join(anns_dir, image_name.split('.')[0] + '.txt')
    with open(txt_name, 'r') as file:
        instance_info = json.load(file)
    
    to_camera = instance_info["to_camera"]
    
    angles = get_true_angle(instance_info["bb3d"])
    
    width, height = instance_info["bb2d"][2], instance_info["bb2d"][3]
    normalize_length = math.sqrt(width**2 + height**2) 
    bb3d = instance_info["bb3d"]
    dimensions = three_normalized_dimensions(bb3d, normalize_length)

    label = list(map(lambda x: str(x), [to_camera] + angles + dimensions))

    return separator.join(label)



wirte_list_image_path("validation")