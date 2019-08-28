import keras
import numpy as np
import json
import _init_paths
from keras.models import load_model

from boxcars_datagen import BoxImageGenerator
from config import output_path, image_dir_test

model_path = '/home/vivacityserver6/repos/BoxCars/cache/snapshots/model_resnet60_adam015.h5'
prediction_saved_path = output_path + '/predictions_file_with_angle.json'
batch_size = 16

estimated_prediction = False

def assigning_evaluation_value(eval_list):
    key_list = ['loss', 'd_loss', 'a0_loss', 'a1_loss', 'a2_loss', 'd_acc', 'a0_acc', 'a1_acc', 'a2_acc']
    return dict(zip(key_list, eval_list))

def evaluation_for_whole_dataset(model, dataset_generator, batch_size):
    test_evaluation = model.evaluate_generator(dataset_generator, steps=dataset_generator.n // batch_size, verbose=1)
    test_evaluation = assigning_evaluation_value(test_evaluation)

    return test_evaluation

def main():
    model = load_model(model_path)

    #loading boxcar generator
    generator_test = BoxImageGenerator("test", batch_size, image_dir_test)

    if estimated_prediction == False:
        evaluation = evaluation_for_whole_dataset(model, generator_test, batch_size)
        print(evaluation)
        with open(prediction_saved_path, 'w+') as json_file:
            json.dump(evaluation, json_file, indent=4)

    else:
        with open(prediction_saved_path, 'r') as json_file:
            predictions = json.load(json_file)
            print(type(predictions))


if __name__ == '__main__':
    main()