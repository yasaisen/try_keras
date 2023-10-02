from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import Sequential

import matplotlib.pyplot as plt

max_features = 10000  # ←考慮作為特徵的文字數量
maxlen = 500  #← 只看每篇評論的前 500 個字


#=====================================================================#


print('讀取資料...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)  # ←載入資料
print(len(x_train), 'train sequences')
print(len(y_test), 'test sequences')

x_train = [x[::-1] for x in x_train]    # ←將訓練資料進行反向順序排列
x_test = [x[::-1] for x in x_test]      # ←將測試資料進行反向順序排列

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)  # ←填補序列資料
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('input_train shape:', x_train.shape)	# shape=(25000, 500)
print('input_test shape:', x_test.shape)	# shape=(25000, 500)


#=====================================================================#


model = Sequential()
model.add(Embedding(max_features, 32))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


#=====================================================================#


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('24_Maccuracy.png')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('24_Mloss.png')

plt.show()







