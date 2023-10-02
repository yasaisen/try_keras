from keras.applications import inception_v3
from keras import backend as K    #匯入 Keras 的後端功能

import tensorflow as tf
import numpy as np

# tf.compat.v1.disable_eager_execution()



    

    # ↓ 我們不會訓練這模型，因此使用此方法中止所有訓練相關的操作
K.set_learning_phase(0)

                                    # ↓ 該模型將載入預先訓練的 ImageNet 權重
model = inception_v3.InceptionV3(weights='imagenet',include_top=False)
                                        # ↑ 此 InceptionV3 神經網路
                                        #       不包含最上層的全連接層，說明如下小編補充。    


    #=====================================================================#


layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
    # ↑ InceptionV3 中層的名稱
}


    #=====================================================================#


# ↓ 建立一個將層名稱 (key) 對應到層物件 (value) 的字典
layer_dict = dict([(layer.name, layer) for layer in model.layers])
# ↓ 用 keres 的後端 (目前為 tensorflow) 建立一個純量變數 (初始值為 0)
loss = K.variable(0.)   # 稍後透過加上層的貢獻度到此純量變數來定義損失

for layer_name in layer_contributions:  
    coeff = layer_contributions[layer_name]     # 取出該層的貢獻係數
    activation = layer_dict[layer_name].output  # 取出該層的啟動函數輸出值 (即輸出張量)
    
    scaling = K.prod(K.cast(K.shape(activation), 'float32')) # 計算輸出張量元素總數做為縮放係數
    

    # ↓ 將各層的輸出張量進行元素平方後相加，
    # 並乘上貢獻度、除以縮放係數後 (L2 norm)，做為損失值。 
    # loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
                                                    # ↑ 為避免處理邊界圖像，
                                                    #     我們只處理非邊界的像素至損失中


    #=====================================================================#




    # dream = model.input     # 設定 V3 模型的輸入
    # print(dream.shape)      # 其輸入格式為彩色圖片 (3-channels) shape = (?,?,?,3)


    # 後端的 gradients() 可以根據 loss 計算 dream 的梯度
    # grads = tf.gradients(loss, dream)[0]    
                                # ↑ 因為我們只輸入一個張量，所以是第一筆的輸出結果






with tf.GradientTape() as gtape:
    loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
    dream = model.input

    print('===============================================================')
    print(loss)
    print(type(loss))
    print('===============================================================')
    print(dream)
    print(type(dream))
    print('===============================================================')
    # print(type(loss.numpy()))
    print('===============================================================')


grads = gtape.gradient(loss, dream)

print(type(grads))

print('===============================================================')




# print(grads.shape)  # 此梯度結果是一個張量，與 dream 大小相同                               

grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # 正規化梯度 (重要技巧)
                                            # 防止除以低於 0.0000001 的值而造成梯度太高

# 在給定輸入圖像的情況下，自訂一個 Keras 函數以取得損失值和梯度值
outputs = [loss, grads]     # 設定輸出串列
fetch_loss_and_grads = K.function([dream], outputs)
# ↑ 函式名稱                # 設定輸入 ↑      # ↑ 設定輸出

# 取得損失及梯度
def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])    # 呼叫函數取得輸出 outs = [損失, 梯度]
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# 執行梯度上升
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations): 
        loss_value, grad_values = eval_loss_and_grads(x)    # 取得損失及梯度
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        # print('...grad value at', i, ':', grad_values)  # 若印出梯度，可以觀察到最後梯度會趨近於 0 
        x += step * grad_values     # 將圖片與梯度進行相加
    return x


#=====================================================================#


# import scipy
# import numpy as np
# from keras.preprocessing import image   # 匯入 Keras image 對圖片進行預處理的功能

# # 對圖片進行 Inception V3 預處理
# def preprocess_image(image_path):
#     img = image.load_img(image_path)    # 根據路徑載入圖片
#     img = image.img_to_array(img)       # 將圖片轉成陣列 array
#     print(img.shape)                    # 例如：(682, 1024, 3)，1024x682 的彩色圖片
#     img = np.expand_dims(img, axis=0)   # 以軸 0 對 array 的 shape 進行擴展
#     print(img.shape)                    # 例如：擴展後的 shape (1, 682, 1024, 3)
#     img = inception_v3.preprocess_input(img) # 將圖像預處理為 Inception V3 可以處理的張量
#     return img

# # 將 Inception V3 所做的預處理進行反向操作，轉回圖片格式
# def deprocess_image(x):
#     if K.image_data_format() == 'channels_first':
#         x = x.reshape((3, x.shape[2], x.shape[3]))
#         x = x.transpose((1, 2, 0))
#     else:
#         x = x.reshape((x.shape[1], x.shape[2], 3))
#     x /= 2.
#     x += 0.5
#     x *= 255.
#     x = np.clip(x, 0, 255).astype('uint8')  # 將數字限制在 0-255 之間
#     return x

# # 進行圖片的比例縮放
# def resize_img(img, size):      
#     img = np.copy(img)
#     # 先設定縮放因子，可以想像原本的 shape (1, width, height, 3) 
#     # 會乘上這個 factor 
#     factors = (1,       
#                float(size[0]) / img.shape[1],
#                float(size[1]) / img.shape[2],
#                1)
#     # 將以樣條插直法的技術對圖片進行縮放，其中 order 為樣條插值法的次數
#     return scipy.ndimage.zoom(img, factors, order=1)    


# # 儲存圖片，儲存前要先反轉 Inception V3 所做的預處理
# def save_img(img, fname):       
#     pil_img = deprocess_image(np.copy(img))     # 反轉
#     scipy.misc.imsave(fname, pil_img)
#     # 由於 imsave() 方法將在 SciPy 1.2.0 時被移除，官方建議我們改用 imageio.imwrite
#     # import imageio
#     # imageio.imwrite(fname, pil_img)


# #=====================================================================#


# step = 0.01             # 每次梯度上升的變化量
# num_octave = 3          # 執行梯度上升的比例數量
# octave_scale = 1.8      # 比例間的大小比率
# iterations = 20         # 每個比例下執行的梯度上升的次數
# max_loss = 10.          # 如果損失大於 10，則中斷梯度上升，以避免產生醜陋的圖像

# base_image_path = 'original_photo_deep_dream.jpg'   # 基底圖片的檔案位置
# img = preprocess_image(base_image_path) # 載入圖片並進行預處理

# original_shape = img.shape[1:3]         # 取得圖片的原比例寬高 (tuple)
# successive_shapes = [original_shape]    # 將不同比例的寬高存入一個 list 中
# for i in range(1, num_octave):
#     shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
#     successive_shapes.append(shape)
# # 請注意!產生的寬高比例為 原比例、縮小 1.4 倍、縮小 1.4x1.4 倍
# successive_shapes = successive_shapes[::-1] # 反轉寬高比例 list，使它們按順序遞增 

# original_img = np.copy(img)
# shrunk_original_img = resize_img(img, successive_shapes[0]) # 產生最小的圖片

# for shape in successive_shapes:     # 開始逐次放大圖片 (八階圖)
#     print('Processing image shape', shape)
#     img = resize_img(img, shape)    # 放大圖片，在這裡就已經損失細節了，注意！第一次
#                                     # 進到迴圈時，img 是由最大變成最小
#     img = gradient_ascent(img,                      # 執行梯度上升，改變 img 圖片
#                           iterations=iterations,
#                           step=step,
#                           max_loss=max_loss)
#     # 將小比例的圖片放大至目前比例，會造成像素顆粒化 (小變大，損失細節)
#     upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
#     # 將原始圖片縮小至目前比例 (大變小，保有細節)
#     same_size_original = resize_img(original_img, shape)
#     # 相減求得損失的細節，例如原有 A+B，放大後只剩 A，(A+B) - (A) = (B) 損失的東西 
#     lost_detail = same_size_original - upscaled_shrunk_original_img
#     img += lost_detail              # 將細節加回圖片中
#     shrunk_original_img = resize_img(original_img, shape)
#     save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

# save_img(img, fname='final_dream.png')





