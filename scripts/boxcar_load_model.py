import keras
import numpy as np
import json
import os
import _init_paths
from keras.models import load_model

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator


model_path = '/home/vivacityserver6/repos/BoxCars/cache/snapshot/model_008.h5'    #need to modify
prediction_saved_path = '/home/vivacityserver6/repos/BoxCars/scripts/predictions_file_with_angle.json'
OUTPUT_PATH = '/home/vivacityserver6/repos/BoxCars/output/'
batch_size = 64
part_size = 64

model = load_model(model_path)
estimated_3DBB = None
estimated_prediction = False

if estimated_3DBB == None:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

dataset.initialize_data('test')

generator_test = BoxCarsDataGenerator(dataset, "test", batch_size, training_mode=False)

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

            
            prediction_ints = {}
            for label in range(len(prediction)):
            
                predictions_one_label =  prediction[label][0].tolist()
                index_most_posible_class = predictions_one_label.index(max(predictions_one_label))

                if label == 0:
                    prediction_ints['output_d'] = index_most_posible_class
                elif label == 1:
                    prediction_ints['output_a'] = index_most_posible_class
            
            if i < 5:
                print(vehicle_id, instance_id, prediction_ints)
            
            predictions[vehicle_id][instance_id] = prediction_ints

        i += 1

    with open(prediction_saved_path, "w") as write_file:
        json.dump(predictions, write_file, separators=(',', ':'), indent = 4)

    return predictions

def visualize_prediction_part(predictions):
    for idx, item in predictions.items():
        if idx < part_size:

            #create a folder to save output
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'boxcar_test_visualize')):
                os.mkdir(os.path.join(OUTPUT_PATH, 'boxcar_test_visualize'))

            
        
