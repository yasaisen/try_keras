from keras.preprocessing.image import load_img, img_to_array
import tensorflow
tensorflow.compat.v1.disable_eager_execution()
target_image_path = 'target.jpg'         # 目標圖片的路徑
style_reference_image_path = 'style.jpg'  # 風格參考圖片的路徑

# 以目標圖片的寬高比來設定生成圖片的高 (400px) 與對應的寬
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)


#=====================================================================#


import numpy as np
from keras.applications import vgg19    # 匯入 VGG19 模型

def preprocess_image(image_path):
    img = load_img(image_path,          # 載入圖片，並以指定的尺寸進行調整，shape = (img_height,img_width,3)
                   target_size=(img_height, img_width))
    img = img_to_array(img)             # 將圖片轉為 array
    img = np.expand_dims(img, axis=0)   # 擴展維度 => shape = (1,img_height,img_width,3)
    img = vgg19.preprocess_input(img)   # 將圖片預處理為 VGG19 可以處理的張量格式 
    return img

def deprocess_image(x):
    # 對各個 channel 加回 ImageNet 的 channel 平均像素值，
    # 這反轉了 vgg19.preprocess_input 預處理所進行的 0 中心化
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]   # 將圖片 channel 從 'BGR' 轉換為 'RGB'。這也是反轉 vgg19.preprocess_input 的預處理
    x = np.clip(x, 0, 255).astype('uint8')  # 將數字限制在 0-255 之間
    return x


#=====================================================================#


from keras import backend as KJ

import tensorflow as K
# from tensorflow.keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

combination_image = KJ.placeholder((1, img_height, img_width, 3))    # 預留位置將用來儲存生成圖片

# 將這三個圖片張量串接成一批次張量
input_tensor = KJ.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# 使用一批三個圖片張量作為輸入以建構 VGG19 神經網路。該模型將載入預先訓練的 ImageNet 權重
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')


#=====================================================================#


                # ↓ 目標圖片  # ↓ 生成圖片
def content_loss(base, combination):
    return KJ.sum(K.square(combination - base))  # 圖片張量元素相減後的平方和


#=====================================================================#


def gram_matrix(x): # 計算格拉姆矩陣，以下矩陣運算請參考程式 8.5 的第 8~11 項說明
    features = KJ.batch_flatten(KJ.permute_dimensions(x, (2, 0, 1)))
    gram = KJ.dot(features, K.transpose(features))
    return gram
              # ↓ 風格參考圖片 # ↓ 生成圖片
def style_loss(style, combination): # 
    S = gram_matrix(style)          # 取得風格參考圖片的格拉姆矩陣
    C = gram_matrix(combination)    # 取得生成圖片的格拉姆矩陣
    channels = 3
    size = img_height * img_width
    return KJ.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))    
            # 格拉姆矩陣元素相減後的平方和


#=====================================================================#


                       # ↓ 生成圖片
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return KJ.sum(K.pow(a + b, 1.25))
                    # 張量次方運算請參考 8.5.py 的第 12 點


#=====================================================================#


# ↓ 建立一個將層名稱 (key) 對應到層物件的啟動函數輸出張量 (value) 的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = 'block5_conv2'  # 要用於內容損失的層名稱
# 要用於風格損失的層名稱
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# 三種損失的加權平均值的權重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025



# import tensorflow as tf
# from tensorflow.keras import backend as K

with K.GradientTape() as gtape:


        
    loss = KJ.variable(0.)   # 透過將所有損失加到此純量變數來定義總損失

    # 計算內容損失並加到總損失中
    layer_features = outputs_dict[content_layer]
    # 0 是因為將三個圖片張量串接輸入到 VGG19 時，目標圖片張量為第 1 個
    target_image_features = layer_features[0, :, :, :]  
    combination_features = layer_features[2, :, :, :]   # 生成圖片為第 3 個

    
    loss = loss + (content_weight * content_loss(target_image_features, combination_features))


    # 計算指定的層所產生的風格損失並加到總損失中                 
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]   # 風格圖片為第二個
        combination_features = layer_features[2, :, :, :]       # 理由同上
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)

    # 計算總變異損失並加到總損失中
    loss += total_variation_weight * total_variation_loss(combination_image)


#=====================================================================#


# 後端的 gradients() 可以根據 loss 計算生成圖片對應的梯度
# grads = K.gradients(loss, combination_image)[0]
                                            # ↑ 因為我們只輸入一個張量，所以是第一筆的輸出結果

# fetch_loss_and_grads = K.function([combination_image], [loss, grads]) # 用於取得目前損失值與梯度值的函數



grads = gtape.gradient(loss, combination_image)
fetch_loss_and_grads = K.function([combination_image], [loss, grads])



# 這個類別以透過兩個單獨方法呼叫的方式，取得損失值和梯度值，
# 並封裝成 fetch_loss_and_grads 方式，這是 SciPy 優化器所要求的
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):                      # 第一次呼叫取得損失值
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])    # 取得損失值與梯度值
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values      # 將梯度值先暫存在記憶體
        return self.loss_value

    def grads(self, x):                     # 將暫存於記憶體的梯度值回傳
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()     # 實例化 Evaluator 物件


#=====================================================================#


from scipy.optimize import fmin_l_bfgs_b    # L-BFGS 優化器
# from scipy.misc import imsave               # 儲存圖片功能
import cv2
import time                                 

result_prefix = 'style_transfer_result'     # 生成圖片的檔名
iterations = 20                             # 迭代總次數                         

x = preprocess_image(target_image_path)     # 這是初始狀態：目標圖片
x = x.flatten()  # 將 array 變成 1 維 (拉平，scipy.optimize.fmin_l_bfgs_b 只能處理平面向量)
print(x.shape) 
for i in range(iterations):
    print(f'第 {i} 次迭代')
    start_time = time.time()           
    # 對生成圖像的像素執行L-BFGS優化以以最小化神經風格損失。
    # 注意到必須將計算損失值的函數與計算梯度值的函數作為兩個單獨的參數傳遞     
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,    # 要最小化的損失函數
                                     x,                         # 初始狀態圖片
                                     fprime=evaluator.grads,    # 梯度函數
                                     maxfun=20)
    print('目前損失值:', min_val)

    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)  
    fname = result_prefix + '_at_iteration_%d.png' % i

    
    # imsave(fname, img)
    cv2.imwrite(fname, img)

    end_time = time.time()
    print(f'圖片以檔名 {fname} 儲存')
    print(f'第 {i} 次迭代所花費的時間:{end_time - start_time}')


