

# from itertools import Predicate
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras import models

import numpy as np 
import matplotlib.pyplot as plt
import cv2

img_path = r'cats_and_dogs_small/test/123.JPG'


VGG16_model = VGG16(weights='imagenet')  # 頂部包含了密集連接的分類器 (預設 include_top=True)


img = image.load_img(img_path, target_size=(224, 224))
img_ado = image.img_to_array(img)  # shape=(224, 224, 3)#將PIL物件轉為float32的Numpy陣列
img_ado = np.expand_dims(img_ado, axis=0)# shape=(1, 224, 224, 3)
img_ado = preprocess_input(img_ado) # 預處理批次量 (這會對每一 channel 做顏色值正規化)


preds = VGG16_model.predict(img_ado)
print('預測結果:', decode_predictions(preds, top=3)[0])

np.argmax(preds[0])

################################################################################

last_conv_layer = VGG16_model.get_layer('block5_conv3') # block5_conv3 層的輸出特徵圖, 其為 VGG16 中的最後一個卷積層
heatmap_model = models.Model([VGG16_model.input], [last_conv_layer.output, VGG16_model.output])

# grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

with tf.GradientTape() as gtape:
    conv_output, Predictions = heatmap_model(img_ado)
    prob = Predictions[:, np.argmax(Predictions[0])]
    grads = gtape.gradient(prob, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)


heatmap = np.maximum(heatmap, 0)

max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10

heatmap /= max_heat

plt.matshow(heatmap[0], cmap='viridis')


# heatmap = np.maximum(heatmap, 0)
# heatmap /= np.max(heatmap)
# plt.matshow(heatmap[0], cmap='viridis')
# plt.show()


img = cv2.imread(img_path)
heatmap_1 = cv2.resize(heatmap[0], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
heatmap_1 = np.uint8(255 * heatmap_1)
heatmap_1 = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)
frame_out = cv2.addWeighted(img, 0.5, heatmap_1, 0.5, 0)
cv2.imwrite('heatP.jpg', frame_out)

plt.figure()
plt.imshow(frame_out)
plt.show()






