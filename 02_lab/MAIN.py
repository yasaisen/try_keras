from tensorflow import keras 

import numpy as np
from keras.backend import relu, sigmoid
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizer_v2 import adam
from tensorflow.keras.optimizers import SGD, RMSprop
#from keras.optimizers import SGD, Adam, RMSprop
from keras import optimizers
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import Regularizer
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import os

def Generator(train_path, validation_path):

    now_path = os.getcwd()

    train_dir = os.path.join(now_path, train_path)
    validation_dir = os.path.join(now_path, validation_path)

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=20,class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=20,class_mode='binary')

    return train_generator, validation_generator

train_path = 'cats_and_dogs_small/train'
validation_path = 'cats_and_dogs_small/validation'

train, validation= Generator(train_path, validation_path)


model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['acc'])

#model.fit(train, y_train, batch_size=64, epochs=5)
history = model.fit_generator(
    train,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation,
    validation_steps=50)

model.save('cats_and_dogs_small_1.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()






