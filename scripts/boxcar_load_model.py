import keras
import numpy as np
import json
import os
import _init_paths
from keras.models import load_model

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

'''
input:  model_path,
        
'''

model_path = '/home/vivacityserver6/repos/BoxCars/cache/snapshot/model_60bins_resnet_008.h5'    #need to modify
output_path = '/home/vivacityserver6/repos/BoxCars/output/'
prediction_saved_path = output_path + 'prediction_60bins_boxcar.json'
saved_prediction = False
batch_size = 64
part_size = 16


def predictions_for_whole_dataset(model, dataset):
    part_data = dataset.split['test']
    predictions = {}
    i = 0
    for vehicle_id, label in part_data:
        instances = dataset.dataset["samples"][vehicle_id]["instances"]  
        predictions[vehicle_id] = {}
        for instance_id in range(len(instances)):
            vehicle, instance, bb3d = dataset.get_vehicle_instance_data(vehicle_id, instance_id)
            image = dataset.get_image(vehicle_id, instance_id)

            image = (image.astype(np.float32) - 116)/128.

            image.resize((1, image.shape[0],image.shape[1], image.shape[2]))
            prediction = model.predict_on_batch(image)

            
            prediction_per_image = {}

            prediction_per_image['output_d'] = prediction_img[0].tolist()[0]
            prediction_per_image['output_a'] = prediction_img[1].tolist()[0]

            if i < 5:
                print(vehicle_id, instance_id, prediction_per_image)
            
            predictions[vehicle_id][instance_id] = prediction_ints

        i += 1

    with open(prediction_saved_path, "w+") as write_file:
        json.dump(predictions, write_file, separators=(',', ':'), indent = 4)

    return predictions

def visualize_prediction_img(prediction_image, image, image_path):
    output_d = prediction_per_image['output_d']
    prediction_per_image['output_a']

def visualize_prediction_batch():
    model = load_model(model_path)
    estimated_3DBB = None
    estimated_prediction = False

    if estimated_3DBB == None:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
    else:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                                use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

    dataset.initialize_data('test')

    part_data = dataset.split['test']

    image_path = os.path.join(output_path, 'boxcar_test_visualize')
    #create a folder to save output
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    if not saved_prediction:
        predictions = predictions_for_whole_dataset(model, dataset)
    else:
        with open(prediction_saved_path, 'r') as file:
            predictions = json.load(file)

    for idx, (vehicle_id, label) in enumerate(part_data):
        if idx < part_size:
            instances = dataset.dataset["samples"][vehicle_id]["instances"]

            for instance_id in range(len(instances)):
                vehicle, instance, bb3d = dataset.get_vehicle_instance_data(vehicle_id, instance_id)
                image = dataset.get_image(vehicle_id, instance_id)
                prediction_image = predictions[vehicle_id][instance_id]
                visualize_prediction_img(prediction_image, image, image_path)

if '__name__' == '__main__':
    visualize_prediction_batch()


                
            

            
        
