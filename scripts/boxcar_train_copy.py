import _init_paths
import os
import json
from utils import ensure_dir, parse_args

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation,Conv2D, MaxPooling2D, BatchNormalization, Flatten, LeakyReLU
from keras.callbacks import ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt

batch_size = 128
epochs = 15
epoch_period = 1
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
model_main.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=input_shape))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

model_main.add(Conv2D(64, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

model_main.add(Conv2D(128, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(64, (1, 1), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(128, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

#----------

model_main.add(Conv2D(256, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(128, (1, 1), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(256, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

#----------

model_main.add(Conv2D(512, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(256, (1, 1), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(256, (1, 1), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())
model_main.add(MaxPooling2D(pool_size=(2, 2)))

#----------
'''
model_main.add(Conv2D(1024, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (1, 1), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(1024, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(512, (1, 1), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
model_main.add(BatchNormalization())

model_main.add(Conv2D(1024, (3, 3), activation='linear'))
model_main.add(LeakyReLU(alpha=0.1))
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
generator_test = BoxCarsDataGenerator(dataset, "test", batch_size = 1, training_mode=False)

#%% callbacks
ensure_dir(tensorboard_dir)
ensure_dir(snapshots_dir)
tb_callback = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
saver_callback = ModelCheckpoint(os.path.join(snapshots_dir, "model_test_{epoch:03d}.h5"), period=3)


print(dataset.X['train'].shape)
print(dataset.X['validation'].shape)
print(generator_train.n)
print(generator_val.n)

current_epoch = 0

output_a_loss = []
output_a_acc = []
val_output_a_loss = []
val_output_a_acc = []
epochs_list = []


for training_loop in range(epochs // epoch_period):
    h = model.fit_generator(generator=generator_train, 
                        steps_per_epoch=generator_train.n // batch_size,
                        epochs=current_epoch + epoch_period,
                        verbose=1,
                        validation_data=generator_val,
                        validation_steps=generator_val.n // batch_size,
                        callbacks=[tb_callback, saver_callback],
                        initial_epoch = current_epoch,
                        )
                        
    history = h.history
    print(history)
    current_epoch += epoch_period
    epochs_list.append(current_epoch)
    output_a_loss.extend(history['output_a_loss'])
    output_a_acc.extend(history['output_a_acc'])
    val_output_a_loss.extend(history['val_output_a_loss'])
    val_output_a_acc.extend(history['val_output_a_acc'])

    plt.plot(epochs_list, output_a_loss, 'r--')
    plt.plot(epochs_list, output_a_acc, 'r-')
    plt.plot(epochs_list, val_output_a_loss, 'b--')
    plt.plot(epochs_list, val_output_a_acc, 'b-')
    plt.xlim(0, epochs)
    plt.legend(['output_a_loss', 'output_a_acc', 'val_output_a_loss', 'val_output_a_acc'])
    plt.xlabel('Epochs')
    plt.ylabel('loss and accuracy')
    plt.savefig('./loss_acc_image.png')
    
    # compute and save loss and accuracy on training set
    # train_evaluation = model.evaluate_generator(generator_train, steps=generator_train.n // batch_size, verbose=1)
    # print('train_eval: ', train_evaluation)
    # train_evaluations[current_epoch] = assigning_evaluation_value(train_evaluation)
    # train_evaluation = model.evaluate_generator(generator_train, steps=generator_train.n // batch_size, verbose=1)
    # print('train_eval: ', train_evaluation)
    # train_evaluations[current_epoch] = assigning_evaluation_value(train_evaluation)
    # # compute and save loss and accuracy on test set
    # test_evaluation = model.evaluate_generator(generator_test, steps=generator_test.n // batch_size, verbose=1)
    # print('test_eval: ', test_evaluation)
    # test_evaluations[current_epoch] = assigning_evaluation_value(test_evaluation)
    # val_evaluation = model.evaluate_generator(generator_val, verbose=1)
    # print('val_eval: ', val_evaluation)
    # val_evaluations[current_epoch] = assigning_evaluation_value(val_evaluation)
'''
total_eval = {}
total_eval['train'] = train_evaluations
total_eval['val'] = val_evaluations
total_eval['test'] = test_evaluations
'''
 
model.save('./model_test_epoch{}_direction_angle.h5'.format(epochs))

with open('./loss_acc.json', 'w') as file:
    json.dump(total_eval, file, separators=(',', ':'), indent = 4)

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

