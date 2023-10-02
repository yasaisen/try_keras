

from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras import models

import numpy as np
import matplotlib.pyplot as plt
import os


img_path = 'cats_and_dogs_small/test/cats/cat.1502.jpg'
model_path = '02_cats_and_dogs_small_2.h5'

now_path = os.getcwd()
img_dir = os.path.join(now_path, img_path)
model_dir = os.path.join(now_path, model_path)

model = models.load_model(model_dir)
model.summary()

img = load_img(img_dir, target_size=(150, 150))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.


layer_inputs = model.input
layer_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs=layer_inputs, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)


layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

