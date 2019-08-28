import _init_paths
import os
import json
from utils import ensure_dir, parse_args
from learningratefinder import LearningRateFinder

from boxcars_datagen import BoxImageGenerator

import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten, LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adadelta, SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, TerminateOnNaN

import matplotlib.pyplot as plt
from config import *

latest_model_path = ""

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.labels = ['loss', 'output_d_loss', 'output_a0_loss', 'output_dim0_loss', 'output_d_acc', 'output_a0_acc', 'output_dim0_acc']
        self.losses = [[] for _ in range(len(self.labels))]

        self.epoch_labels = ['loss', 'val_loss', 'val_output_d_loss', 'val_output_a0_loss', 'val_output_dim0_loss', 'val_output_d_acc', 'val_output_a0_acc', 'val_output_dim0_acc']
        self.epoch_logs = [[] for _ in range(len(self.epoch_labels))]

        self.batch_period = 100
        self.period_loss = []
        self.epoch_number = 0
        self.steps = 852

    # def on_batch_begin(self, batch, logs={}):
    #     with open('./record_vehicle_id.txt', 'a+') as file:
    #         file.write('batch number: ' + str(batch) + '\n')


    def on_batch_end(self, batch, logs={}):
        for idx, loss in enumerate(self.losses):
            loss.append(logs.get(self.labels[idx]))
        
        batch_list = list(range(batch + self.epoch_number*self.steps + 1))

        plt.plot(batch_list, self.losses[0], 'r-', label=self.labels[0], linewidth=2)

        plt.xlabel("batches")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(loss_fig)
        plt.clf()

        plt.plot(batch_list, self.losses[1], 'b-', label=self.labels[1], linewidth=1)
        plt.plot(batch_list, self.losses[2], 'k-', label=self.labels[2], linewidth=1)
        plt.plot(batch_list, self.losses[3], 'y-', label=self.labels[3], linewidth=1)
            
        plt.xlabel("batches")
        plt.ylabel("Losses")
        plt.legend()

        plt.savefig(losses_fig)
        plt.clf()      

        plt.plot(batch_list, self.losses[4], 'b--', label=self.labels[4], linewidth=1)
        plt.plot(batch_list, self.losses[5], 'k--', label=self.labels[5], linewidth=1)
        plt.plot(batch_list, self.losses[6], 'y--', label=self.labels[6], linewidth=1)
            
        plt.xlabel("batches")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig(acc_fig)
        plt.clf()      

        # # save best model
        # if (batch + 1) % self.batch_period == 0:
        #     self.period_loss.append(logs.get('loss'))

        #     if batch == self.batch_period -1 or self.period_loss[-1] <= min(self.period_loss[:-1]):
        #         self.model.save_weights("./model_60bins_resnet_SGD_temp.h5")
        #         stored_train_loss = {'batch_list': batch_list}
        #         for i in range(1, 4):
        #             stored_train_loss[self.labels[i]] = list(map(lambda x: float(x), self.losses[i]))
        #         with open('./train_loss_history.json', 'w+') as file:
        #             json.dump(stored_train_loss, file, indent=4)
        

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_number = epoch

    def on_epoch_end(self, epoch, logs={}):
        epoch_history = {'epoch': epoch, 'logs': logs}
        with open(output_path + '/train_history.json', 'a+') as file:
            json.dump(epoch_history, file, indent=4)
        
        for idx, loss in enumerate(self.epoch_logs):
            loss.append(logs.get(self.epoch_labels[idx]))

        epoch_list = list(range(epoch + 1))
        logs.get(self.epoch_labels[idx])

        plt.plot(epoch_list, self.epoch_logs[0], 'r-', label=self.epoch_labels[0], linewidth=2)
        plt.plot(epoch_list, self.epoch_logs[1], 'r--', label=self.epoch_labels[1], linewidth=1)

        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(val_loss_fig)
        plt.clf()

        for idx, linestyle in zip([2, 3, 4, 5, 6, 7], ['b-', 'k-', 'y-', 'b--', 'k--', 'y--']):
            plt.plot(epoch_list, self.epoch_logs[idx], linestyle, label=self.epoch_labels[idx], linewidth=1)

              
        plt.xlabel("epochs")
        plt.ylabel("Accuracy and loss")
        plt.legend()

        plt.savefig(val_losses_fig)
        plt.clf()      

        with open(output_path + '/learning_rate.json', 'a+') as file:
            file.write(str(K.get_value(self.model.optimizer.lr)) + '\n')

        if (epoch+1) % 15 == 0:
            print('epoch: ', epoch, 'changing learning rate')
            current_lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, current_lr * 10)

        

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

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def extra_layer(x):
    y = resnet_layer(x)
    y = resnet_layer(y)
    y = resnet_layer(y)
    return Flatten()(y)


snapshots_dir = os.path.join(cache, "snapshots")
tensorboard_dir = os.path.join(cache, "tensorboard")
loss_history = LossHistory()
###build training model

if using_VGG:
    main_input = Input(shape = input_shape, name='main_input')
    model_main = VGG_model()
    x = model_main(main_input)
elif using_resnet:
    model_main = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = model_main.output
    flat_x = Flatten()(x)
    
direction_output = Dense(direction_number, activation = 'softmax', name='output_d')(flat_x)

# angle_0_output_previous = Dense(500, activation="relu")(x)

angle_0_output = Dense(angle_bin_number, activation = 'softmax', name='output_a0')(flat_x)
angle_1_output = Dense(angle_bin_number, activation = 'softmax', name='output_a1')(flat_x)
angle_2_output = Dense(angle_bin_number, activation = 'softmax', name='output_a2')(flat_x)

dimension_0_output = Dense(dimension_bin_number, activation = 'softmax', name='output_dim0')(extra_layer(x))
dimension_1_output = Dense(dimension_bin_number, activation = 'softmax', name='output_dim1')(extra_layer(x))
dimension_2_output = Dense(dimension_bin_number, activation = 'softmax', name='output_dim2')(extra_layer(x))

output_list = [direction_output, 
               angle_0_output, 
               angle_1_output, 
               angle_2_output,
               dimension_0_output,
               dimension_1_output,
               dimension_2_output]

if using_VGG:
    model = Model(inputs=main_input, outputs=output_list)

elif using_resnet:
    model = Model(inputs=model_main.input, outputs=output_list)

# optimizer = Adadelta(rho = 0.95)
# optimizer = SGD(lr=1e-5, momentum=0.9)
# model.compile(loss=keras.losses.sparse_categorical_crossentropy,
#               optimizer=optimizer,
#               metrics=['accuracy'])
from keras.optimizers import Adam
optimizer=Adam(lr=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

###training

generator_train = BoxImageGenerator(datamode="train", batch_size=batch_size, image_dir=image_dir_train, training_mode=True)
generator_val = BoxImageGenerator(datamode="validation", batch_size=batch_size, image_dir=image_dir_val, training_mode=False)

#%% callbacks
ensure_dir(tensorboard_dir)
ensure_dir(snapshots_dir)
#tb_callback = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)
saver_callback = ModelCheckpoint(os.path.join(snapshots_dir, snapshots_file), period=5)
terminate = TerminateOnNaN()


if lr_search:
    old_model = load_model(latest_model_path)
    lrf = LearningRateFinder(model)
    lrf.find(
        generator_train,
        1e-8, 1e+1,
        epochs=1,
        stepsPerEpoch=len(generator_train),
        batchSize=48
    )

else:
    # if continue_train:
    #     model.load_weights('./model_60bins_resnet_SGD_temp.h5')
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! generator_train", generator_train.n)
    h = model.fit_generator(generator=generator_train, 
                        steps_per_epoch= len(generator_train),
                        epochs=epochs,
                        verbose=1,
                        validation_data=generator_val,
                        validation_steps=len(generator_val),
                        callbacks=[loss_history, saver_callback, terminate],
                        initial_epoch = 0, use_multiprocessing=False
                        )
                            
    # history = h.history

    model.save('./model_angle_and_dimension_60bins_resnet_epoch{}.h5'.format(epochs))

# with open('./loss_acc_6bins_resnet.json', 'w') as file:
#     json.dump(total_eval, file, separators=(',', ':'), indent = 4)

