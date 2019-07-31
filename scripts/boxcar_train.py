import _init_paths
import os
from utils import ensure_dir, parse_args

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard


batch_size = 8
epochs = 1
class_number = 2
input_shape = (224, 224, 3)
estimated_3DBB = None
cache = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "cache"))

if estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

dataset.initialize_data("train")
dataset.initialize_data("validation")

generator_train = BoxCarsDataGenerator(dataset, "train", batch_size, training_mode=True)
generator_val = BoxCarsDataGenerator(dataset, "validation", batch_size, training_mode=False)

iterator = iter(generator_val)
for item in iterator:
    
print(iterator.check_repeatation)

'''
output_final_model_path = os.path.join(cache, "final_model.h5")
snapshots_dir = os.path.join(cache, "snapshots")
tensorboard_dir = os.path.join(cache, "tensorboard")

###build training model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(class_number, activation='softmax'))


model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

###training

#initialize dataset for training
dataset.initialize_data("train")
dataset.initialize_data("validation")

generator_train = BoxCarsDataGenerator(dataset, "train", batch_size, training_mode=True)
generator_val = BoxCarsDataGenerator(dataset, "validation", batch_size, training_mode=False)

#%% callbacks
ensure_dir(tensorboard_dir)
ensure_dir(snapshots_dir)
tb_callback = TensorBoard(tensorboard_dir, histogram_freq=1, write_graph=False, write_images=False)
saver_callback = ModelCheckpoint(os.path.join(snapshots_dir, "model_{epoch:03d}_{val_acc:.2f}.h5"), period=4 )


print(dataset.X['train'].shape)
print(dataset.X['validation'].shape)
print(generator_train.n)
print(generator_val.n)

model.fit_generator(generator=generator_train, 
                    samples_per_epoch=generator_train.n,
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=generator_val,
                    nb_val_samples=generator_val.n,
                    )

model.save('./model.h5')

#%% evaluate the model 
print("Running evaluation...")
dataset.initialize_data('test')
generator_test = BoxCarsDataGenerator(dataset, "test", batch_size = 1, training_mode=False, generate_y=False)
#print(generator_test.n)

predictions = model.predict_generator(generator_test, generator_test.n)
#predictions = model.predict_generator(generator_test, 100)
print(predictions.shape)

print(" -- Accuracy: %.2f%%"%(single_acc*100))
print(" -- Track accuracy: %.2f%%"%(tracks_acc*100))
'''