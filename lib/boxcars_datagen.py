import os
import sys
import cv2
import numpy as np
import math
# import keras
from utils import cross_from_points, get_true_angle, three_normalized_dimensions, image_preprocess
from boxcars_image_transformations import alter_HSV, add_bb_noise_flip

script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(script_dir, '..', 'scripts')))


from config import BOXCARS_LIST_TRAIN, BOXCARS_LIST_VAL, BOXCARS_LIST_TEST
# 
class BoxImageGenerator():
    def __init__(self, datamode, batch_size, image_dir ,image_size=(224, 224)):
        self.batch_size = batch_size
        self.separator = ' '
        self.image_dir = image_dir

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
        self.__load_list_path(list_image_file)

    def __len__():
        return int(math.floor(len(self.x) / float(self.batch_size)))

    def __load_list_path(self, list_image_file):
        with open(list_image_file) as file:
            self.list_paths = file.read().splitlines()

    def generate_batch_data(self):
        indexes = np.arange(len(self.list_image_file))
        np.random.shuffle(indexes)

        for i in range(len(indexes) // self.batch_size):

            x = np.empty([self.batch_size] + list(self.image_size) + [3], dtype=np.float32)
            batch_y = {}
            for idx in range(1, 8):
                batch_y['y{}'.format(idx)] = np.empty([self.batch_size] + [1], dtype=np.float32)


            for j in range(self.batch_size):
                image_sample_str = self.list_paths[i * self.batch_size + j]
                image_path = os.path.join(self.image_dir, image_sample_str.split(self.separator)[0])
                image = cv2.imread(image_path)
                if self.datamode = "train:
                    image = alter_HSV(image)
                    bb_noise = None
                    flip = bool(random.getrandbits(1)) # random flip
                    image, bb3d = add_bb_noise_flip(image, bb3d, flip, bb_noise)
                x[j, ...] = image

                for idx in range(1, 8):
                    batch_y['y{}'.format(idx)][j, ...] = image_sample_str.split(self.separator)[idx]

            yield(x, {'output_d': batch_y['y1'], 
                    'output_a0': batch_y['y2'], 
                    'output_a1': batch_y['y3'], 
                    'output_a2': batch_y['y4'],
                    'output_dim0': batch_y['y5'],
                    'output_dim1': batch_y['y6'],
                    'output_dim2': batch_y['y7']})


