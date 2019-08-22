import _init_paths
import os
import json
import cv2
import numpy as np
from box_dataset import draw_bb2d, draw_bb3d
from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator


import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sns
import numpy as np


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

if not os.path.exists('./bb2d_visualize/'):
    os.mkdir('./bb2d_visualize/')

def find_shape_list(part_data):

    two_d_width = []
    two_d_height = []
    vehicle_id, instance_id = 2271, 0
    # for idx, (vehicle_id, label) in enumerate(part_data):
    instances = dataset.dataset["samples"][vehicle_id]["instances"]
    #     for instance_id in range(len(instances)):
    image = dataset.get_image(vehicle_id, instance_id)
    bb2d = instances[instance_id]["2DBB"]
    vehicle, instance, bb3d  = dataset.get_vehicle_instance_data(vehicle_id, instance_id )

    # if idx >= 1000 and idx <2000:
    image, c = draw_bb2d(image, bb2d)
    image = draw_bb3d(image, bb3d)
    cv2.imwrite('./img_{}_{}.png'.format(vehicle_id, instance_id ), image) # bb2d_visualize
    cv2.imwrite('./imgtest2_{}_{}.png'.format(vehicle_id, instance_id ), image[bb2d[1]:bb2d[1] + bb2d[3], bb2d[0]:bb2d[0] + bb2d[2], :]) # bb2d_visualize

    #         two_d_width.append(bb2d[2])
    #         two_d_height.append(bb2d[3])

    # return two_d_width, two_d_height

            # if image.any() == None:
            #     print('None image', vehicle_id, instance_id)
            # elif np.shape(image) == ():
            #     print('shape(image) == ()', vehicle_id, instance_id)
            # elif np.sum(image) == 0:
            #     print('black image', vehicle_id, instance_id)

# image_width, image_height = find_shape_list(part_data)
# sns.set_context("talk")
# fig = plt.figure(figsize=(10, 10))
# fig.clf()

# ax = fig.add_subplot(111)

# plt.scatter(image_width, image_height, marker='.', color='mediumslateblue', alpha=1)
# ax.set_xlabel("width (pixel)")
# ax.set_ylabel("height (pixel)")

# plt.savefig("./two_d_box_scatter.png")
# fig.clf()


# im = dataset.get_image(5044, 2)
# cv2.imwrite('./example_after_processing.png', im)
    
find_shape_list(part_data)