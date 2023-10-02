
#from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.imagenet_utils import decode_predictions
#from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array


import numpy as np
import matplotlib.pyplot as plt
import os


def Generator(test_path):
    now_path = os.getcwd()

    test_dir = os.path.join(now_path, test_path)

    #test_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return test_generator

model_path = '04_cats_and_dogs_small_2.h5'
test_path = 'cats_and_dogs_small/test'
img_path = 'cats_and_dogs_small/test/cats/cat.1510.jpg'

# test = Generator(test_path)

img = load_img(img_path, target_size=(150, 150))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
#img_tensor /= 255.
img_tensor = preprocess_input(img_tensor)

model = models.load_model(model_path)


# result = model.evaluate_generator(test, steps=50)
# print ('\nTest Acc : ', result[1])

preds = model.predict(img_tensor)
#print ('\nresult : ', preds)
print('預測結果:', decode_predictions(preds))
#np.argmax(preds[0])
