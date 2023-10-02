#from keras.layers import Embedding

from keras.datasets import imdb
from keras import preprocessing

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


#embedding_layer = Embedding(1000, 64)  # 建立嵌入向量層至少須指定兩個參數：可能的 tokens 數量 (此處為 1, 000, 最少是 1+最大文字索引) 和嵌入向量的維數 (此處為 64)

#=====================================================================#

max_features = 10000  # ←設定作為特徵的文字數量
maxlen = 20 # ←在 20 個文字之後切掉文字資料 (在 max_features 最常見的文字中)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)  # ←將資料以整數 lists 載入
print(x_train.shape)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)  # ←將整數 lists 轉換為 2D 整數張量, 形狀為(樣本數 samples, 最大長度 maxlen)
print(x_train.shape)
print(x_train[0])
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#=====================================================================#

# 1. 指定嵌入向量層的最大輸入長度, 以便之後可以攤平嵌入向量的輸入。在嵌入向量層之後, 啟動函數輸出的 shape 為 (樣本數 samples, 最大長度 maxlen, 8）
# 2. 將嵌入向量的 3D 張量展平為 2D 張量, 形狀為(樣本數 samples, 最大長度 maxlen * 8)

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen)) # ←1...

model.add(Flatten()) # ← 2...

model.add(Dense(1, activation='sigmoid')) # ← 在頂部加上分類器


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, 
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

