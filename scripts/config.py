# -*- coding: utf-8 -*-
import os
#%%
# change this to your location
BOXCARS_DATASET_ROOT = "/home/vivacityserver6/datasets/BoxCars116k/" 

#%%
BOXCARS_IMAGES_ROOT = os.path.join(BOXCARS_DATASET_ROOT, "images")
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")

BOXCARS_LIST_TRAIN = os.path.join(BOXCARS_DATASET_ROOT, "list", "training.txt")
BOXCARS_LIST_TEST = os.path.join(BOXCARS_DATASET_ROOT, "list", "test.txt")
BOXCARS_LIST_VAL = os.path.join(BOXCARS_DATASET_ROOT, "list", "validation.txt") 

batch_size = 48
epochs = 40
direction_number = 2
angle_bin_number = 60
angle_number = 3
dimension_bin_number = 60
dimension_number = 3
input_shape = (224, 224, 3)
estimated_3DBB = None
using_VGG = False
using_resnet = True
continue_train = False

cache = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "cache"))
snapshots_file = "model_resnet60_adam_extra_layer{epoch:03d}.h5"

output_path = "./output_classic_setting"
if not os.path.exists(output_path):
    os.mkdir(output_path)
angle_fig_file = output_path + "/loss_acc_3angles_60bins_resnet.png"
dimension_fig_file = output_path + "/loss_acc_3dimesnions_60bins_resnet.png"

loss_fig = output_path + "/loss_60bins_resnet_adam.png"
losses_fig = output_path + "/losses_60bins_resnet_adam.png"
acc_fig = output_path + "/acc_60bins_resnet_adam.png"
val_loss_fig = output_path + "/valloss_60bins_resnet_adam.png"
val_losses_fig = output_path + "/vallosses_60bins_resnet_adam.png"

lr_search = False

image_dir_train = '/home/vivacityserver6/datasets/BoxCars116k/ims_train'
anns_dir_train = '/home/vivacityserver6/datasets/BoxCars116k/anns_train'
image_dir_val = '/home/vivacityserver6/datasets/BoxCars116k/ims_val'
image_dir_test = '/home/vivacityserver6/datasets/BoxCars116k/ims_test'