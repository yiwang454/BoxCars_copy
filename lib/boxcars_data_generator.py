# -*- coding: utf-8 -*-
import cv2
import numpy as np
from keras.preprocessing.image import Iterator
from boxcars_image_transformations import alter_HSV, image_drop, unpack_3DBB, add_bb_noise_flip
import random

#%%
class BoxCarsDataGenerator(Iterator):
    def __init__(self, dataset, part, batch_size=8, training_mode=False, seed=None, generate_y = True, image_size = (224, 224)):
        assert image_size == (224, 224), "only images 224x224 are supported by unpack_3DBB for now, if necessary it can be changed"  #temporary change for image size 
        assert dataset.X[part] is not None, "load some classification split first"
        super().__init__(dataset.X[part].shape[0], batch_size, training_mode, seed)
        self.part = part
        self.generate_y = generate_y
        self.dataset = dataset
        self.image_size = image_size
        self.training_mode = training_mode
        self.check_repeatation = {}
        if self.dataset.atlas is None:
            self.dataset.load_atlas()        

    #%%
    def next(self):
        with self.lock:
            index_array = next(self.index_generator) #, current_index, current_batch_size
        
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        x = np.empty([self.batch_size] + list(self.image_size) + [3], dtype=np.float32)
        y1 = np.empty([self.batch_size] + [1], dtype=np.float32)
        y2 = np.empty([self.batch_size] + [1], dtype=np.float32)
        y3 = np.empty([self.batch_size] + [1], dtype=np.float32)
        y4 = np.empty([self.batch_size] + [1], dtype=np.float32)
        y5 = np.empty([self.batch_size] + [1], dtype=np.float32)
        y6 = np.empty([self.batch_size] + [1], dtype=np.float32)
        y7 = np.empty([self.batch_size] + [1], dtype=np.float32)

        for i, ind in enumerate(index_array):
            vehicle_id, instance_id = self.dataset.X[self.part][ind]
            '''
            #might be used to check how image data has been shuffled and fed in
            if vehicle_id in self.check_repeatation.keys():
                if instance_id in self.check_repeatation[vehicle_id].keys():
                    self.check_repeatation[vehicle_id][instance_id] += 1
                else:
                    self.check_repeatation[vehicle_id][instance_id] = 0
            else:
                self.check_repeatation[vehicle_id] = {}
            '''
            vehicle, instance, bb3d = self.dataset.get_vehicle_instance_data(vehicle_id, instance_id)
            image = self.dataset.get_image(vehicle_id, instance_id)
            if self.training_mode:
                image = alter_HSV(image) # randomly alternate color
                image = image_drop(image) # randomly remove part of the image
                bb_noise = np.clip(np.random.randn(2) * 1.5, -5, 5) # generate random bounding box movement
                flip = bool(random.getrandbits(1)) # random flip
                image, bb3d = add_bb_noise_flip(image, bb3d, flip, bb_noise) 
            image = (image.astype(np.float32) - 116)/128.
            x[i, ...] = image
            y1[i, ...] = self.dataset.Y[self.part][ind][0]
            y2[i, ...] = self.dataset.Y[self.part][ind][1]
            y3[i, ...] = self.dataset.Y[self.part][ind][2]
            y4[i, ...] = self.dataset.Y[self.part][ind][3]
            y5[i, ...] = self.dataset.Y[self.part][ind][4]
            y6[i, ...] = self.dataset.Y[self.part][ind][5]
            y7[i, ...] = self.dataset.Y[self.part][ind][6]

        if not self.generate_y:
            return x

        return (x, {'output_d': y1, 
                    'output_a0': y2, 
                    'output_a1': y3, 
                    'output_a2': y4,
                    'output_dim0': y5,
                    'output_dim1': y6,
                    'output_dim2': y7})

