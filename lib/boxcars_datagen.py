import os
import sys
import cv2
import numpy as np
import random
import math
import keras
# import keras
from utils import cross_from_points, get_true_angle, three_normalized_dimensions, image_preprocess
from boxcars_image_transformations import alter_HSV, add_bb_noise_flip

script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(script_dir, '..', 'scripts')))

from keras.utils import to_categorical
from config import BOXCARS_LIST_TRAIN, BOXCARS_LIST_VAL, BOXCARS_LIST_TEST

class BoxImageGenerator(keras.utils.Sequence):
    def __init__(self, datamode, batch_size, image_dir ,image_size=(224, 224), training_mode=False, shuffle = True):
        self.batch_size = batch_size
        self.separator = ' '
        self.image_dir = image_dir
        self.training_mode = training_mode
        self.shuffle = shuffle

        if datamode == "train":
            self.list_image_file = BOXCARS_LIST_TRAIN
        elif datamode == "validation":
            self.list_image_file = BOXCARS_LIST_VAL
        elif datamode == "test":
            self.list_image_file = BOXCARS_LIST_TEST
        else:
            raise Exception('Setting a wrong data mode, should be either training. validation or test')

        self.datamode = datamode

        self.image_size = image_size

        self.list_paths = None
        self.__load_list_path(self.list_image_file)

        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        self.list_paths = None
        self.__load_list_path(self.list_image_file)

    def __len__(self):
        return int(math.floor(len(self.list_paths) / float(self.batch_size)))

    def __load_list_path(self, list_image_file):
        with open(list_image_file) as file:
            self.list_paths = file.read().splitlines()

    def __getitem__(self, index):
        local_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        x = np.empty([self.batch_size] + list(self.image_size) + [3], dtype=np.float32)
        batch_y = {}
        for idx in range(1, 8):
            batch_y['y{}'.format(idx)] = np.empty([self.batch_size] + [1], dtype=np.int32)

        for j in range(len(local_indexes)):
            image_sample_str = self.list_paths[local_indexes[j]]
            image_path = os.path.join(self.image_dir, image_sample_str.split(self.separator)[0])
            image = image_preprocess(cv2.imread(image_path))
            # print(image.shape)

            # if self.datamode == "train":
            #     image = alter_HSV(image)
            #     # print(image.shape)

            #     bb_noise = None
            #     flip = bool(random.getrandbits(1)) # random flip
            #     # image = add_bb_noise_flip(image, bb3d, flip, bb_noise)
            
            # print(image.shape)

            x[j, ...] = image

            for idx in range(1, 8):
                if idx == 1:
                    batch_y['y{}'.format(idx)][j, ...] = int(image_sample_str.split(self.separator)[idx])
                else:
                    batch_y['y{}'.format(idx)][j, ...] = int(image_sample_str.split(self.separator)[idx]) - 1 
        # print(batch_y['y5'])

        return x, {'output_d': to_categorical(batch_y['y1'], num_classes=2), 
                'output_a0': to_categorical(batch_y['y2'], num_classes=60), 
                'output_a1': to_categorical(batch_y['y3'], num_classes=60), 
                'output_a2': to_categorical(batch_y['y4'], num_classes=60),
                'output_dim0': to_categorical(batch_y['y5'], num_classes=60),
                'output_dim1': to_categorical(batch_y['y6'], num_classes=60),
                'output_dim2': to_categorical(batch_y['y7'], num_classes=60)}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)





