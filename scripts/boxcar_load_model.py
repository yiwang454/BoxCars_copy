import keras
import numpy as np
import json
import os
import _init_paths
from math import sin, cos, pi
import cv2

from keras.models import load_model

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator
from utils import cross_from_points, get_true_angle



'''
input:  model_path,    
'''

model_path = '/home/vivacityserver6/repos/BoxCars/cache/snapshots/model_60bins_resnet_008.h5'    #need to modify
output_path = '/home/vivacityserver6/repos/BoxCars/output/'
prediction_saved_path = output_path + 'prediction_60bins_boxcar.json'
saved_prediction = True
batch_size = 64
part_size = 16
FONT = cv2.FONT_HERSHEY_SIMPLEX

def from_angle_to_coordinates(angle, img_size):
    y = img_size[0] // 2
    x = img_size[1] // 2
    left_point = (x, y)
    right_point = (round(x + 60 * cos( - angle * pi / 180)), round(y + 60 * sin( - angle * pi / 180)))
    return left_point, right_point

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
            prediction_img = model.predict_on_batch(image)

            
            prediction_per_image = {}

            prediction_per_image['output_d'] = prediction_img[0].tolist()[0]
            prediction_per_image['output_a'] = prediction_img[1].tolist()[0]

            if i < 5:
                print(vehicle_id, instance_id, prediction_per_image)
            
            predictions[vehicle_id][instance_id] = prediction_per_image
        i += 1

    with open(prediction_saved_path, "w+") as write_file:
        json.dump(predictions, write_file, separators=(',', ':'), indent = 4)

    return predictions

def visualize_prediction_img(prediction_per_image, image, image_name, bb3d):
    directions_predictions = prediction_per_image['output_d']
    direction = directions_predictions.index(max(directions_predictions))
    angle_predictions = prediction_per_image['output_a']
    angle = (angle_predictions.index(max(angle_predictions)) - 30) * 3.0
    true_angle = get_true_angle(bb3d)

    L, R = from_angle_to_coordinates(angle, image.shape)
    true_L, true_R = from_angle_to_coordinates(true_angle, image.shape)
    
    cv2.arrowedLine(image, true_L, true_R, (255, 0, 0), 3)
    cv2.putText(image, 'predict_angle: {}, true_angle: {}'.format(angle, true_angle), (0, 20), FONT, 3, (255,0,0), 1, cv2.LINE_AA)
    
    if direction == 1:
        text = 'to_camera'
        cv2.arrowedLine(image, L, R, (0,0,255),3)	# draw green box for 2D bbox
        cv2.putText(image, text , (0, image.shape[0]), FONT, 3, (0,0,255), 1, cv2.LINE_AA)
    else:
        text = 'from_camera'
        cv2.arrowedLine(image, L, R, (0,255,0),3)	# draw green box for 2D bbox
        cv2.putText(image, text ,(0, image.shape[0]), FONT, 3, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(image_name, image)

def visualize_prediction_batch(getting_angle):
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

    image_path = os.path.join(output_path, 'boxcar_test_visualize_5000_resnet50_60bins')
    #create a folder to save output
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    if not saved_prediction:
        predictions = predictions_for_whole_dataset(model, dataset)
    else:
        with open(prediction_saved_path, 'r') as file:
            predictions = json.load(file)


    alpha, beta, gamma, predicts = [], [], [], []
    angles = [alpha, beta, gamma, predicts]

    for idx, (vehicle_id, label) in enumerate(part_data):
        #if idx < 5000 + part_size and idx >= 5000:
        #if idx < 5000 + part_size and idx >= 5000:
        if getting_angle:
            instances = dataset.dataset["samples"][vehicle_id]["instances"]

            for instance_id in range(len(instances)):
                vehicle, instance, bb3d = dataset.get_vehicle_instance_data(vehicle_id, instance_id)
                if saved_prediction:
                    prediction_image = predictions[str(vehicle_id)][str(instance_id)]
                else:
                    prediction_image = predictions[vehicle_id][instance_id]
                # image_name = os.path.join(image_path, '{}_{}.png'.format(vehicle_id, instance_id))
                # image = dataset.get_image(vehicle_id, instance_id)
                # visualize_prediction_img(prediction_image, image, image_name, bb3d)

                #The following lines: added for getting angle
                angle_predictions = prediction_image['output_a']
                predict_angle = (angle_predictions.index(max(angle_predictions)) - 30) * 3.0
                true_angles = cross_from_points(bb3d)
                true_angles.append(predict_angle)
                for idx, angle in enumerate(angles):
                    angle.append(true_angles[idx])

    return angles

def main():
    visualize_prediction_batch(True)

if __name__ == '__main__':
    main()

                
            

            
        
