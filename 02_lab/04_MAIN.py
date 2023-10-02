

from keras.backend import relu, sigmoid
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizer_v2 import adam, rmsprop
from keras import optimizer_v2
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import Regularizer
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from tensorflow import optimizers

import numpy as np
import matplotlib.pyplot as plt
import os

def Generator(train_path, validation_path, test_path):

    now_path = os.getcwd()

    train_dir = os.path.join(now_path, train_path)
    validation_dir = os.path.join(now_path, validation_path)
    test_dir = os.path.join(now_path, test_path)

    #train_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, )

    #validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    #test_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return train_generator, validation_generator, test_generator

###################  SETUP HERE!!!  ###################
train_path = 'cats_and_dogs_small/train'
validation_path = 'cats_and_dogs_small/validation'
test_path = 'cats_and_dogs_small/test'
codecode = '04'
###################  SETUP HERE!!!  ###################

train, validation, test= Generator(train_path, validation_path, test_path)


conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))

model = Sequential()

model.add(conv_base)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr = 1e-5), metrics=['acc'])
#model.fit(train, y_train, batch_size=64, epochs=5)
history = model.fit_generator(
    train,
    steps_per_epoch=100,
    epochs=100,#################################################
    validation_data=validation,
    validation_steps=50)

model.save(codecode + '_cats_and_dogs_small_2.h5')


result = model.evaluate_generator(test, steps=50)
print ('\nTest Acc : ', result[1])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(codecode + '_Figure_1.png')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(codecode + '_Figure_2.png')
plt.show()






