import _init_paths
import os
import json
import numpy as np
from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

import keras


batch_size = 1
input_shape = (224, 224, 3)
estimated_3DBB = None


if estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split='hard', load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split='hard', load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)


dataset.initialize_data("train")
dataset.initialize_data("validation")

part_data = dataset.split['train']

generator_train = BoxCarsDataGenerator(dataset, "train", batch_size, training_mode=True)
print(generator_train.n)

# for vehicle_id, label in enumerate(part_data):
#     instances = dataset.dataset["samples"][vehicle_id]["instances"]
#     for instance_id in range(len(instances)):
#         image = dataset.get_image(vehicle_id, instance_id)
#         if image.any() == None:
#             print('None image', vehicle_id, instance_id)
#         elif np.shape(image) == ():
#             print('shape(image) == ()', vehicle_id, instance_id)
#         elif np.sum(image) == 0:
#             print('black image', vehicle_id, instance_id)




    
