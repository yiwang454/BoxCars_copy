import keras
import numpy as np
import json
import _init_paths
from keras.models import load_model

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

model_path = '/home/vivacityserver6/repos/BoxCars/cache/snapshots/model_6bins_resnet_003.h5'
prediction_saved_path = '/home/vivacityserver6/repos/BoxCars/scripts/predictions_file_with_angle.json'
batch_size = 64

estimated_3DBB = None
estimated_prediction = False

def assigning_evaluation_value(eval_list):
    key_list = ['loss', 'd_loss', 'a_loss', 'd_acc', 'a_acc']
    return dict(zip(key_list, eval_list))

def evaluation_for_whole_dataset(model, dataset_generator, batch_size):
    test_evaluation = model.evaluate_generator(dataset_generator, steps=dataset_generator.n // batch_size, verbose=1)
    test_evaluation = assigning_evaluation_value(test_evaluation)

    return test_evaluation

def main():
    model = load_model(model_path)

    if estimated_3DBB == None:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
    else:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                                use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

    dataset.initialize_data('test')

    generator_test = BoxCarsDataGenerator(dataset, "test", batch_size, training_mode=False)

    if estimated_prediction == False:
        evaluation = evaluation_for_whole_dataset(model, generator_test, batch_size)
        print(evaluation)

    else:
        with open(prediction_saved_path, 'r') as json_file:
            predictions = json.load(json_file)
            print(type(predictions))


if __name__ == '__main__':
    main()