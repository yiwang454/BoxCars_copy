import _init_paths
import os
import json
from utils import ensure_dir, parse_args

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation,Conv2D, MaxPooling2D, BatchNormalization, Flatten, LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt

batch_size = 64
epochs = 15
epoch_period = 1
direction_number = 2
angle_bin_number = 60
angle_number = 3
input_shape = (224, 224, 3)
estimated_3DBB = None
using_VGG = False
using_resnet = True
cache = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "cache"))
snapshots_file = "model_3angles_60bins_resnet_{epoch:03d}.h5"
fig_file = "./loss_acc_3angles_60bins_resnet.png"

def VGG_model():
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
    
    # model_main.add(Conv2D(1024, (3, 3), activation='linear'))
    # model_main.add(LeakyReLU(alpha=0.1))
    # model_main.add(BatchNormalization())

    # model_main.add(Conv2D(512, (1, 1), activation='linear'))
    # model_main.add(LeakyReLU(alpha=0.1))
    # model_main.add(BatchNormalization())

    # model_main.add(Conv2D(1024, (3, 3), activation='linear'))
    # model_main.add(LeakyReLU(alpha=0.1))
    # model_main.add(BatchNormalization())

    # model_main.add(Conv2D(512, (1, 1), activation='linear'))
    # model_main.add(LeakyReLU(alpha=0.1))
    # model_main.add(BatchNormalization())

    # model_main.add(Conv2D(1024, (3, 3), activation='linear'))
    # model_main.add(LeakyReLU(alpha=0.1))
    # model_main.add(BatchNormalization())
    return model_main

if estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split='hard', load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split='hard', load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = estimated_3DBB)

snapshots_dir = os.path.join(cache, "snapshots")
tensorboard_dir = os.path.join(cache, "tensorboard")

###build training model



if using_VGG:
    main_input = Input(shape = input_shape, name='main_input')
    model_main = VGG_model()
    x = model_main(main_input)
elif using_resnet:
    model_main = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = Flatten()(model_main.output)
    
direction_output = Dense(direction_number, activation = 'softmax', name='output_d')(x)

angle_0_output = Dense(angle_bin_number, activation = 'softmax', name='output_a0')(x)
angle_1_output = Dense(angle_bin_number, activation = 'softmax', name='output_a1')(x)
angle_2_output = Dense(angle_bin_number, activation = 'softmax', name='output_a2')(x)

if using_VGG:
    model = Model(inputs=main_input, outputs=[direction_output, angle_0_output, angle_1_output, angle_2_output])

elif using_resnet:
    model = Model(inputs=model_main.input, outputs=[direction_output, angle_0_output, angle_1_output, angle_2_output])

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
tb_callback = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
saver_callback = ModelCheckpoint(os.path.join(snapshots_dir, snapshots_file), period=2)

current_epoch = 0

output_angle_loss = [[] for _ in range(angle_number)]
output_angle_acc = [[] for _ in range(angle_number)]
val_output_angle_loss = [[] for _ in range(angle_number)]
val_output_angle_acc = [[] for _ in range(angle_number)]

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
    for i, angle_loss in enumerate(output_angle_loss):
        key = 'output_a{}_loss'.format(i)
        angle_loss.extend(history[key])

    for i, angle_acc in enumerate(output_angle_acc):
        key = 'output_a{}_acc'.format(i)
        angle_acc.extend(history[key])

    for i, val_angle_loss in enumerate(val_output_angle_loss):
        key = 'val_output_a{}_loss'.format(i)
        val_angle_loss.extend(history[key])

    for i, val_angle_acc in enumerate(val_output_angle_acc):
        key = 'val_output_a{}_acc'.format(i)
        val_angle_acc.extend(history[key])

    for i in range(angle_number):
        plt.subplot(1, angle_number, i + 1)
        plt.plot(epochs_list, output_angle_loss[i], 'r--')
        plt.plot(epochs_list, output_angle_acc[i], 'r-')
        plt.plot(epochs_list, val_output_angle_loss[i], 'b--')
        plt.plot(epochs_list, val_output_angle_acc[i], 'b-')
        plt.xlim(0, epochs)
        plt.legend(['output_a{}_loss'.format(i), 'output_a{}_acc'.format(i), 'val_output_a{}_loss'.format(i), 'val_output_a{}_acc'.format(i)])
        plt.xlabel('Epochs')
        plt.ylabel('loss and accuracy')
    plt.savefig(fig_file)
        

'''
total_eval = {}
total_eval['train'] = train_evaluations
total_eval['val'] = val_evaluations
total_eval['test'] = test_evaluations
'''
 
model.save('./model_3angles_60bins_resnet_epoch{}_direction_angle.h5'.format(epochs))

'''
with open('./loss_acc_6bins_resnet.json', 'w') as file:
    json.dump(total_eval, file, separators=(',', ':'), indent = 4)


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

