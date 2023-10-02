from keras.models import Sequential, Model
from keras import layers, Input

seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))					   #1...
seq_model.add(layers.Dense(10, activation='softmax'))

input_tensor = Input(shape=(64,))   #← 建立一個初始張量

# 將初始張量傳入 Dense 層得到輸出張量 x
x = layers.Dense(32, activation='relu')(input_tensor)
 
# 再將第一層的結果 x 傳入第 2 個 Dense 層得到輸出張量 y                2...
y = layers.Dense(32, activation='relu')(x) 

# 再將第二層的結果 y 傳入最後一個 Dense 層得到最後的輸出張量 output_tensor
output_tensor = layers.Dense(10, activation='softmax')(y) 

# Model 類別 "用" 初始的輸入張量和最後的輸出張量來得到模型物件
model = Model(input_tensor, output_tensor)
model.summary()    