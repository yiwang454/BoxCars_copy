import cv2
import os
import json

import keras
import numpy as np
from keras.models import load_model

# should be modified to args. later
OUTPUT_PATH = '/home/vivacityserver6/repos/BoxCars/output'
MODEL_PATH = '/home/vivacityserver6/repos/BoxCars/cache/snapshots/model_60bins_resnet_008.h5'
folder_path = '/home/vivacityserver6/repos/BoxCars/output/cropped_img'

def read_img_name(file_path):
    # read img name from a file with all img needs to be classified
    img_names = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            img_names.append(line.strip('\n'))
        
    return img_names

def save_labeled_img(prediction, image_name):
    
    # save cropped image with direction labels
    if prediction[0] >= prediction[1]:
        label_image_name = image_name + 'from_camera.jpg'
    else:
        label_image_name = image_name + 'to_camera.jpg'

    cv2.imwrite(os.path.join(IMG_PATH, 'labeled_img' , label_image_name), image) # no more labeled_img folder no
    
def save_one_img_prediction(img_name, predictions):
    if not os.path.exists(PREDICTION_PATH):
        os.mkdir(PREDICTION_PATH)

    with open(os.path.join(FILE_FOLDER, img_name) + '.txt', 'w') as file:
        file.write(predictions)


def predictions_for_folder(model, folder_path):
    '''
    input: path of the trained model, a txt file that contain the name of all images
    return: prediction of direction of the car: 
            in a dictionary with the form of predictions[image_name]['from_camera']

    aim input: path of the trained model, one image
    aim return: prediction result in the term of 
                predictions = {'output_d': , 'output_a': }
    '''
    prediction_folder = {}
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        # read and preprocess img
        image_pro = (image.astype(np.float32) - 116)/128.
        image_predict = cv2.resize(image_pro, (224, 224), interpolation=cv2.INTER_AREA)
        image_predict.resize((1, image_predict.shape[0],image_predict.shape[1], image_predict.shape[2]))

        prediction_img = model.predict_on_batch(image_predict)
        prediction_folder[file.split('.')[0]] = {}
        prediction_folder[file.split('.')[0]]['output_d'] = prediction_img[0].tolist()[0]
        prediction_folder[file.split('.')[0]]['output_a'] = prediction_img[1].tolist()[0]

    return prediction_folder
    '''
    prediction_ints = {}
    for label in range(len(prediction)):
            
        predictions_one_label =  prediction[label][0].tolist()
        index_most_posible_class = predictions_one_label.index(max(predictions_one_label))

        if label == 0:
            prediction_ints['output_d'] = index_most_posible_class
        elif label == 1:
            prediction_ints['output_a'] = index_most_posible_class

    return prediction_ints
    '''
    

if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    predictions = predictions_for_folder(model, folder_path)
    with open('/home/vivacityserver6/repos/BoxCars/output/prediction_60bins.json', 'w+') as file:
        json.dump(predictions, file, separators=(',', ':'), indent = 4)

    

