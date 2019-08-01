import cv2
import os

import keras
import numpy as np
from keras.models import load_model

# should be modified to args. later
OUTPUT_PATH = '/home/vivacityserver6/repos/BoxCars/output'
PREDICTION_PATH= os.path.join(IMG_PATH, '/predicted_label_files/')
MODEL_PATH = '/home/vivacityserver6/repos/BoxCars/model.h5'

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


def predictions_for_img(model, image, img_name):
    '''
    input: path of the trained model, a txt file that contain the name of all images
    return: prediction of direction of the car: 
            in a dictionary with the form of predictions[image_name]['from_camera']

    aim input: path of the trained model, one image, image_name
    aim return: prediction result in the term of 
                predictions = {'output_d': , 'output_a': }
    '''

    # read and preprocess img
    image_pro = (image.astype(np.float32) - 116)/128.
    image_predict = cv2.resize(image_pro, (224, 224), interpolation=cv2.INTER_AREA)
    image_predict.resize((1, image_predict.shape[0],image_predict.shape[1], image_predict.shape[2]))

    print(model.predict_on_batch(image_predict))

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
    predictions_for_whole_dataset(MODEL_PATH, FILE_PATH)
'''