# -*- coding: utf-8 -*-
import _init_paths
# this should be soon to prevent tensorflow initialization with -h parameter
from utils import ensure_dir, parse_args
args = parse_args(["ResNet50", "VGG16", "VGG19", "InceptionV3"])

# other imports
import os
import time
import sys
import numpy
import json
import codecs

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

#%% initialize dataset
if args.estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
    use_estimated_3DBB = True, estimated_3DBB_path = args.estimated_3DBB)

#%% get optional path to load model
model = None
for path in [args.eval, args.resume]:
    if path is not None:
        print("Loading model from %s"%path)
        model = load_model(path)
        break

print("Model name: %s"%(model.name))
if args.estimated_3DBB is not None and "estimated3DBB" not in model.name:
    print("ERROR: using estimated 3DBBs with model trained on original 3DBBs")
    sys.exit(1)
if args.estimated_3DBB is None and "estimated3DBB" in model.name:
    print("ERROR: using model trained on estimated 3DBBs and running on original 3DBBs")
    sys.exit(1)

args.output_final_model_path = os.path.join(args.cache, model.name, "final_model.h5")
args.snapshots_dir = os.path.join(args.cache, model.name, "snapshots")
args.tensorboard_dir = os.path.join(args.cache, model.name, "tensorboard")


#%% evaluate the model 
print("Running evaluation...")
dataset.initialize_data("test")
generator_test = BoxCarsDataGenerator(dataset, "test", args.batch_size, training_mode=False, generate_y=False)
start_time = time.time()
predictions = model.predict_generator(generator_test, generator_test.n)
end_time = time.time()

np_array_to_list = predictions.tolist()
json_file = "predictions_2.json" 
json.dump(np_array_to_list, codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)

#single_acc, tracks_acc = dataset.evaluate(predictions)
#print(" -- Accuracy: %.2f%%"%(single_acc*100))
#print(" -- Track accuracy: %.2f%%"%(tracks_acc*100))
print(" -- Image processing time: %.1fms"%((end_time-start_time)/generator_test.n*1000))