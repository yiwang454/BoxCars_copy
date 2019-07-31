import _init_paths
import os
from utils import ensure_dir, parse_args

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation,Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard


batch_size = 128
epochs = 4
direction_number = 2
angle_bin_number = 60
input_shape = (224, 224, 3)
estimated_3DBB = None
cache = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "cache"))

if estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split='hard', load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split='hard', load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

output_final_model_path = os.path.join(cache, "final_model.h5")
snapshots_dir = os.path.join(cache, "snapshots")
tensorboard_dir = os.path.join(cache, "tensorboard")

###build training model

main_input = Input(shape = input_shape, name='main_input')

model_main = Sequential()
model_main.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

model_main.add(Conv2D(64, (3, 3), activation='relu'))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

model_main.add(Conv2D(128, (3, 3), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(64, (1, 1), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(128, (3, 3), activation='relu'))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

#----------

model_main.add(Conv2D(256, (3, 3), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(128, (1, 1), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(256, (3, 3), activation='relu'))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

#----------

model_main.add(Conv2D(512, (3, 3), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(256, (1, 1), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (3, 3), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(256, (1, 1), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (3, 3), activation='relu'))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

#----------
'''
model_main.add(Conv2D(1024, (3, 3), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (1, 1), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(1024, (3, 3), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (1, 1), activation='relu'))
model_main.add(BatchNormalization())

model_main.add(Conv2D(1024, (3, 3), activation='relu'))
model_main.add(BatchNormalization())
'''
model_main.add(Flatten())

x = model_main(main_input)
direction_output = Dense(direction_number, activation = 'softmax', name='output_d')(x)
angle_output = Dense(angle_bin_number, activation = 'softmax', name='output_a')(x)
model = Model(inputs=main_input, outputs=[direction_output, angle_output])

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

###training

#initialize dataset for training
dataset.initialize_data("train")
dataset.initialize_data("validation")

generator_train = BoxCarsDataGenerator(dataset, "train", batch_size, training_mode=True)
generator_val = BoxCarsDataGenerator(dataset, "validation", batch_size, training_mode=False)

#initialize dataset for testing
dataset.initialize_data('test')
generator_test = BoxCarsDataGenerator(dataset, "test", batch_size = 1, training_mode=False, generate_y=False)

#%% callbacks
ensure_dir(tensorboard_dir)
ensure_dir(snapshots_dir)
tb_callback = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
saver_callback = ModelCheckpoint(os.path.join(snapshots_dir, "model3_{epoch:03d}.h5"), period=2)


print(dataset.X['train'].shape)
print(dataset.X['validation'].shape)
print(generator_train.n)
print(generator_val.n)

initial_epoch = 0
for training_loop in range(epochs):
    model.fit_generator(generator=generator_train, 
                        steps_per_epoch=generator_train.n // batch_size,
                        epochs=initial_epoch + 2,
                        verbose=1,
                        validation_data=generator_val,
                        validation_steps=generator_val.n // batch_size,
                        callbacks=[tb_callback, saver_callback],
                        initial_epoch = initial_epoch,
                        )
    initial_epoch += 2
    
    # compute and save loss and accuracy on training set
    
    # compute and save loss and accuracy on test set
    predictions = model.predict_generator(generator_test, generator_test.n)
    acc_d, acc_a = dataset.evaluate(predictions)
    
    print('epochs: {}, acc_d: {}, acc_a: {}'.format(initial_epoch, acc_d, acc_a))

model.save('./model3_epoch{}_direction_angle.h5'.format(epochs))

'''
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