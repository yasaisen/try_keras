

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras import models
from tensorflow import gradients

import numpy as np 
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet',include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output 

#with tf.GradientTape() as gtape:
loss = K.mean(layer_output[:, :, :, filter_index]) # 定義損失函數張量, 其為層輸出張量數值取平均
grads = gradients(loss, model.input)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)# 在做除法之前先加上 1e-5 以避免意外地除以 0
iterate = K.function([model.input], [loss, grads]) # 定義一個 Keras 後端函式


loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
#輸出損失張量與梯度張量				↑將此做為輸入張量


# 從帶有雜訊的灰階圖像開始
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128. # 1...

step = 1. # 每個梯度更新的大小
for i in range(40): # 執行梯度上升 40 步
	loss_value, grads_value = iterate([input_img_data]) # 計算損失值和梯度值
	input_img_data += grads_value * step # 2. 以朝向最大化損失調整輸入圖像 (以前SGD 是用 -= 算符, 現在反過來是用 += 算符)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)				# 1. 張量正規化：以 0 為中心, 確保 std 為 0.1 
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1) # 修正成 [0, 1], 即 0-1 之間 
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')		# 2.轉換成 RGB 陣列
    return x


def generate_pattern(layer_name, filter_index, size=150):
	layer_output = model.get_layer(layer_name).output # 取得指定層的輸出張量
	loss = K.mean(layer_output[:, :, :, filter_index]) # 1. 取得指定過濾器的輸出張量, 並以最大化此張量的均值做為損失


	grads = K.gradients(loss, model.input)[0] # 根據此損失計算輸入影像的梯度

	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # 標準化技巧：梯度標準化

	iterate = K.function([model.input], [loss, grads]) # 2.建立 Keras function 來針對給定的輸入影像回傳損失和梯度

	input_img_data = np.random.random((1, size, size, 3)) * 20 + 128. # 3. 從帶有雜訊的灰階影像開始
	

	step = 1.
	for i in range(40): # 執行梯度上升 40 步
		loss_value, grads_value = iterate([input_img_data]) # 4. 針對給定的輸入影像回傳損失和梯度
		input_img_data += grads_value * step

	img = input_img_data[0]
	return deprocess_image(img)	  # 進行圖像後處理後回傳


plt.imshow(generate_pattern('block3_conv1', 0)) # 我們來看看 block3_conv1 層中的過濾器 0 的特徵圖


# for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
#     size = 64
#     margin = 5

#     # 1. 用於儲存結果的空(黑色)影像
#     results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

#     for i in range(8):  # ← 迭代產生網格的行
#         for j in range(8):  # ←迭代產生網格的列
#             # 在 layer_name 中產生過濾器 i +(j * 8) 的 pattern
#             filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

#             # 將結果放在結果網格的方形(i, j)中
#             horizontal_start = i * size + i * margin
#             horizontal_end = horizontal_start + size
#             vertical_start = j * size + j * margin
#             vertical_end = vertical_start + size
#             results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

#     # 顯示網格結果
#     plt.figure(figsize=(20, 20))
#     plt.imshow(results)
#     plt.show()
