import _init_paths
import os
import json
from utils import ensure_dir, parse_args
from learningratefinder import LearningRateFinder

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation,Conv2D, MaxPooling2D, BatchNormalization, Flatten, LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adadelta, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback

import matplotlib.pyplot as plt


batch_size = 64
epochs = 15
epoch_period = 1
direction_number = 2
angle_bin_number = 60
angle_number = 3
dimension_bin_number = 60
dimension_number = 3
input_shape = (224, 224, 3)
estimated_3DBB = None
using_VGG = False
using_resnet = True
cache = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "cache"))
snapshots_file = "model_3angles_and_dimensions_60bins_resnet_SGD_{epoch:03d}.h5"
latest_model_path = ""
angle_fig_file = "./loss_acc_3angles_60bins_resnet.png"
dimension_fig_file = "./loss_acc_3dimesnions_60bins_resnet.png"
lr_search = False

if estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

dataset.initialize_data("train")
dataset.initialize_data("validation")

generator_train = BoxCarsDataGenerator(dataset, "train", batch_size, training_mode=True)
generator_val = BoxCarsDataGenerator(dataset, "validation", batch_size, training_mode=False)


output_final_model_path = os.path.join(cache, "final_model.h5")
snapshots_dir = os.path.join(cache, "snapshots")
tensorboard_dir = os.path.join(cache, "tensorboard")

###build training model

class Training():
    def __init__(self, initial_batch, epochs, epoch_period, angle_number, angle_bin_number, dimension_number, dimension_bin_number, )
model.add(Dense(class_number, activation='softmax'))


model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

###training

#initialize dataset for training
dataset.initialize_data("train")
dataset.initialize_data("validation")

generator_train = BoxCarsDataGenerator(dataset, "train", batch_size, training_mode=True)
generator_val = BoxCarsDataGenerator(dataset, "validation", batch_size, training_mode=False)

#%% callbacks
ensure_dir(tensorboard_dir)
ensure_dir(snapshots_dir)
tb_callback = TensorBoard(tensorboard_dir, histogram_freq=1, write_graph=False, write_images=False)
saver_callback = ModelCheckpoint(os.path.join(snapshots_dir, "model_{epoch:03d}_{val_acc:.2f}.h5"), period=4 )


print(dataset.X['train'].shape)
print(dataset.X['validation'].shape)
print(generator_train.n)
print(generator_val.n)

model.fit_generator(generator=generator_train, 
                    samples_per_epoch=generator_train.n,
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=generator_val,
                    nb_val_samples=generator_val.n,
                    )

model.save('./model.h5')

