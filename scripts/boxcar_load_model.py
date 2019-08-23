from keras.models import load_model
import numpy as np
import json
import os
import _init_paths
from math import sin, cos, pi, sqrt
import cv2
import time
from multiprocessing import Pool
from boxcars_datagen import BoxImageGenerator


from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator
from utils import cross_from_points, get_true_angle, get_angle_from_prediction, visualize_prediction_boxes, visualize_prediction_arrows, three_normalized_dimensions

'''
input:  model_path,    
'''

model_path = '/home/vivacityserver6/repos/BoxCars/model_try_new_gen001.h5'    #need to modify
output_path = '/home/vivacityserver6/repos/BoxCars/output/'
prediction_saved_path = output_path + 'boxcar_fullmodel_debug_again.json'
# vehicle_id_saved_path = output_path + 'vehicle_id.json'
angle_numbers = 60
initial_image_idx = 0
saved_prediction = False
saved_vehicle_id = False
batch_size = 64
part_size = 32
FONT = cv2.FONT_HERSHEY_SIMPLEX

estimated_3DBB = None

if estimated_3DBB == None:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                            use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

image_dir_train = '/home/vivacityserver6/datasets/BoxCars116k/ims_train'
image_dir_val = '/home/vivacityserver6/datasets/BoxCars116k/ims_val'


def predictions_for_whole_dataset(model, datagenerator, tra_generator):
    predictions = model.predict_generator(datagenerator, verbose=1)
    predictions_train = model.predict_generator(tra_generator, verbose=1)

    for i, prediction in enumerate(predictions):
        for j in range(prediction.shape[0]):
            for k in range(prediction.shape[1]):
                if abs(prediction[j][k] - predictions_train[i][j][k] > 0.001):
                    print('error occur', prediction[j][k], predictions_train[i][j][k])

    '''
    running result:
    639/639 [==============================] - 89s 139ms/step
    639/639 [==============================] - 87s 135ms/step
    root@bd787c25d055:/home/vivacityserver6/repos/BoxCars# 
    which means no error occur
    '''

    # if os.path.exists(prediction_saved_path):
    #     os.remove(prediction_saved_path)

    # with open(prediction_saved_path, "a+") as write_file:
    #     for prediction in predictions:
    #         json.dump(prediction.tolist(), write_file, separators=(',', ':'), indent = 4)

    return predictions

def visualize_prediction_batch(getting_angle, angle_idx, model=None, arrow_visualize=False):
    estimated_3DBB = None
    estimated_prediction = False

    if estimated_3DBB == None:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
    else:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                                use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

    dataset.initialize_data('test')

    part_data = dataset.split['test']

    image_path = os.path.join(output_path, 'boxcar_test_visualize_3angles{}_resnet_60bins_{}_arrow_{}'.format(angle_idx, initial_image_idx, arrow_visualize))
    #create a folder to save output
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    if not saved_prediction:
        predictions = predictions_for_whole_dataset(model, dataset)
    else:
        with open(prediction_saved_path, 'r') as file:
            predictions = json.load(file)

    if getting_angle:
        ground_truth, predicts = [], []
        angles = [ground_truth, predicts]

    for idx, (vehicle_id, label) in enumerate(part_data):
        if getting_angle or (not getting_angle and idx < part_size + initial_image_idx and idx >= initial_image_idx):
        
            instances = dataset.dataset["samples"][vehicle_id]["instances"]

            for instance_id in range(len(instances)):
                vehicle, instance, bb3d = dataset.get_vehicle_instance_data(vehicle_id, instance_id)
                if saved_prediction:
                    prediction_image = predictions[str(vehicle_id)][str(instance_id)]
                else:
                    prediction_image = predictions[vehicle_id][instance_id]
                if not getting_angle:
                    image_name = os.path.join(image_path, '{}_{}.png'.format(vehicle_id, instance_id))
                    image = dataset.get_image(vehicle_id, instance_id)
                    if arrow_visualize:
                        visualize_prediction_arrows(prediction_image, image, image_name, bb3d, angle_idx)
                    else:
                        image = visualize_prediction_boxes(prediction_image, image)
                        cv2.imwrite(image_name, image)
                #The following lines: added for getting angle
                else:
                    predict_angle = get_angle_from_prediction(prediction_image, angle_idx)
                    true_angles = cross_from_points(bb3d)
                    several_angles = [- true_angles[angle_idx], predict_angle]
                    for index, angle in enumerate(angles):
                        angle.append(several_angles[index])

    if getting_angle:
        return angles


def get_length_thread(id):
    (vehicle_id, instance_id) = id
    
    vehicle, instance, bb3d = dataset.get_vehicle_instance_data(vehicle_id, instance_id)
    bb2d = instance["2DBB"]
    diagonal_length = sqrt(bb2d[2] ** 2 + bb2d[3] ** 2)
    return three_normalized_dimensions(bb3d=bb3d, normal_length = diagonal_length)


def get_length():
    part_data = dataset.split['train']
    with open('./only_image_id.txt', 'r') as file:
        filtered_list = file.read().splitlines()

    if saved_vehicle_id:
        with open(vehicle_id_saved_path, 'r') as json_file:
            vehicle_instance_id_list = json.load(json_file)

    else:
        vehicle_instance_id_list = [] # should be a list of tuple of (vehicle_id, instance_id)
        for vehicle_id, label in part_data:
            instances = dataset.dataset["samples"][vehicle_id]["instances"]
            for i in range(len(instances)):
                if str(vehicle_id) + '_' +  str(i) in filtered_list:
                    vehicle_instance_id_list.append((vehicle_id, i))
                    
        with open(vehicle_id_saved_path, "w+") as file:
            json.dump(vehicle_instance_id_list, file, indent = 4)
        

    with Pool(8) as p:
        length_info = p.map(get_length_thread, vehicle_instance_id_list)

    rearranged_info = [[] for _ in range(3)]
    for info in length_info:
        for i in range(3):
            rearranged_info[i].append(info[i])

    return rearranged_info

def main():
    # visualize_prediction_batch(False, 0)
    train_generator = BoxImageGenerator('validation', 64, image_dir=image_dir_val, shuffle=False)
    train_generator_2 = BoxImageGenerator('validation', 64, image_dir=image_dir_val, training_mode = True, shuffle=False)

    '''
    no error occur both when setting datamode to be training or validation
    '''

    model = load_model(model_path)
    predictions_for_whole_dataset(model, train_generator, train_generator_2)

if __name__ == '__main__':
    main()

                
            

            
        
