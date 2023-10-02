from keras import layers
# from keras import applications
from tensorflow.keras.applications import Xception
from keras import Input

# 我們使用 Xception 神經網路的卷積基底 (不包含最上層的分類器) 進行影像的特徵萃取
xception_base = Xception(weights=None, include_top=False)

# 建立左、右輸入張量 (左、右鏡頭影像), 其 shape 為 (250, 250, 3), 即為 250x250 的彩色影像。
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

# 呼叫相同的視覺模型兩次, 也就是將影像張量傳入 Xception 神經網路物件。
left_features = xception_base(left_input)
right_features = xception_base(right_input)

# 萃取出的左、右影像特徵張量 shape = (?, 8, 8, 2048)
print(left_features.shape)
print(right_features.shape)

# 串接左右影像特徵張量, shape = (?, 8, 8, 4096)
merged_features = layers.concatenate([left_features, right_features], axis=-1)
print(merged_features.shape)
