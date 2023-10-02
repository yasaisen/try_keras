from tensorflow import keras 

import numpy as np
from keras.backend import relu, sigmoid
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizer_v2 import adam
from tensorflow.keras.optimizers import SGD
#from keras.optimizer import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import Regularizer
from keras import regularizers

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    #
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    #
    return(x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()


model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=5)

result = model.evaluate(x_train, y_train, batch_size=10000)
print ('\nTrain Acc:', result[1])

result = model.evaluate(x_test, y_test, batch_size=10000)
print ('\nTest Acc:', result[1])





