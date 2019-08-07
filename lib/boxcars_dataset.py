# -*- coding: utf-8 -*-
from config import BOXCARS_DATASET,BOXCARS_ATLAS,BOXCARS_CLASSIFICATION_SPLITS
from utils import load_cache, cross_from_points, get_true_angle
import _init_paths

import cv2
import numpy as np

#%%
class BoxCarsDataset(object):
    def __init__(self, load_atlas = False, load_split = None, use_estimated_3DBB = False, estimated_3DBB_path = None):
        self.dataset = load_cache(BOXCARS_DATASET)
        self.use_estimated_3DBB = use_estimated_3DBB
        
        self.atlas = None
        self.split = None
        self.split_name = None
        self.estimated_3DBB = None
        self.X = {}
        self.Y = {}
        for part in ("train", "validation", "test"):
            self.X[part] = None
            self.Y[part] = None # for labels as array of 0-1 flags
            
        if load_atlas:
            self.load_atlas()
        if load_split is not None:
            self.load_classification_split(load_split)
        if self.use_estimated_3DBB:
            self.estimated_3DBB = load_cache(estimated_3DBB_path)
        
    #%%
    def load_atlas(self):
        self.atlas = load_cache(BOXCARS_ATLAS)
    
    #%%
    def load_classification_split(self, split_name):
        self.split = load_cache(BOXCARS_CLASSIFICATION_SPLITS)[split_name]
        self.split_name = split_name
       
    #%%
    def get_image(self, vehicle_id, instance_id):
        """
        returns decoded image from atlas in RGB channel order
        """
        #image = cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
    #%%
    def get_vehicle_instance_data(self, vehicle_id, instance_id, original_image_coordinates=False):
        """
        original_image_coordinates: the 3DBB coordinates are in the original image space
                                    to convert them into cropped image space, it is necessary to subtract instance["3DBB_offset"]
                                    which is done if this parameter is False. 
        """
        vehicle = self.dataset["samples"][vehicle_id]
        instance = vehicle["instances"][instance_id]
        if not self.use_estimated_3DBB:
            bb3d = self.dataset["samples"][vehicle_id]["instances"][instance_id]["3DBB"]
        else:
            bb3d = self.estimated_3DBB[vehicle_id][instance_id]
            
        if not original_image_coordinates:
            bb3d = bb3d - instance["3DBB_offset"]

        return vehicle, instance, bb3d 
            
       
    #%%
    def initialize_data(self, part):
        assert self.split is not None, "load classification split first"
        assert part in self.X, "unknown part -- use: train, validation, test"
        assert self.X[part] is None, "part %s was already initialized"%part
        data = self.split[part]
        x, y = [], []
        for vehicle_id, label in data:
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"])
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])
            in_out = self.dataset["samples"][vehicle_id]['to_camera']
            if in_out:    #to camera is true: vehicle going out
                label_inout = 1
            else:
                label_inout = 0


            #angle as label
            for i in range(num_instances):
                instance = self.dataset["samples"][vehicle_id]["instances"][i]
                bb3d = instance["3DBB"]
                angles = get_true_angle(bb3d)
                y.append([label_inout] + angles)           
                        
        self.X[part] = np.asarray(x,dtype=int)

        y = np.asarray(y,dtype=int)

        '''
        # initialize label when using angle as label
        y_categorical = np.zeros((y.shape[0], 61))  # 60 classes: -90 degree to +90 deree divided into 60 bins
        y_categorical[np.arange(y.shape[0]), y + 30] = 1 # class index starts from 0, angle starts from -90/3
        '''
        self.Y[part] = y
        


    def get_number_of_classes(self):
        return len(self.split["types_mapping"])
        
        
    def evaluate(self, predictions, part="test", top_k=1):

        samples = self.X[part]

        # use with full dats
        #assert samples.shape[0] == predictions.shape[0]            

        part_data = self.split[part]
        
        hits_directions = []
        hits_angles = []
        #hits_tracks = []

        for vehicle_id, label in part_data:

            # GROUND TRUTH OF DIRECTIONS
            in_out = self.dataset["samples"][vehicle_id]['to_camera']
            if in_out:    #to camera is true: vehicle going out
                label_inout = 1
            else:
                label_inout = 0
            print("label_inout", label_inout)

            #############

            # how well prediction worked over the track
            # hits_tracks.append(get_hit(np.mean(predictions[index_of_sample, :], axis=0), label_inout))
            # p.mean(predictions[index_of_sample, :], axis=0), label_inout
            # hits_tracks.append(get_hit())

            #GROUND TRUTH OF ANGLES
            for instance_id in range(len(self.dataset["samples"][vehicle_id]["instances"])):

                instance = self.dataset["samples"][vehicle_id]["instances"][instance_id]
                bb3d = instance["3DBB"]
                angles = get_true_angle(bb3d)

                # how well prediction worked on sample by sample basis
                prediction_d = int(predictions[vehicle_id][instance_id]['output_d']) #need to modify according to new predictions data structure
                prediction_a0 = int(predictions[vehicle_id][instance_id]['output_a0'])
                prediction_a1 = int(predictions[vehicle_id][instance_id]['output_a1'])
                prediction_a2 = int(predictions[vehicle_id][instance_id]['output_a2'])


                if prediction_d == label_inout:
                    hits_directions.append(1.0)
                else:
                    hits_directions.append(0.0)

                if prediction_a0 == angles[0]:
                    hits_angles.append(1.0)
                else:
                    hits_angles.append(0.0)

        #need to add a return about angle
        return np.mean(hits_directions), np.mean(hits_angles) #, np.mean(hits_tracks)
        