from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
import matplotlib.pyplot as plt

#=====================================================================#

imdb_dir = os.path.join(os.getcwd(), 'aclImdb')
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding = 'utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

#=====================================================================#

# texts = []
# labels = []

# with open('11_FileManager_texts.txt', 'r') as f:           #找到或創建
#     data = f.readlines()
#     for data_line in data:
#             texts.append(str(data_line))
#     f.close()

# with open('11_FileManager_labels.txt', 'r') as f:          #找到或創建
#     data = f.readlines()
#     for data_line in data:
#             labels.append(int(data_line))
#     f.close()

#=====================================================================#

maxlen = 100                                               # 100 個文字後切斷評論 (只看評論的前 100 個字)
training_samples = 200                                     # 以 200 個樣本進行訓練
validation_samples = 10000                                 # 以 10, 000 個樣本進行驗證
max_words = 10000                                          # 僅考慮資料集中的前 10, 000 個單詞

tokenizer = Tokenizer(num_words=max_words)                 # 建立一個分詞器, 設定上僅考慮 10000 個最常用的文字 (token), 也就是只會看初始資料的前 10000 個文字
tokenizer.fit_on_texts(texts)                              # 建立文字對應的索引值, 一樣依出現順序來決定, 0 一樣不使用
sequences = tokenizer.texts_to_sequences(texts)            # 將文字轉成整數 list 的序列資料

word_index = tokenizer.word_index
# print(word_index)
print('共使用了 %s 個 token 字詞.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)             # 只取每個評論的前 100 個字 (多切少補) 作為資料張量
labels = np.asarray(labels)                                # 將標籤 list 轉為 Numpy array (標籤張量)

print('資料張量 shape:', data.shape)                        # (25000, 100)
print('標籤張量 shape:', labels.shape)                      # (25000,)

indices = np.arange(data.shape[0])                         # 將資料拆分為訓練集和驗證集, 但首先要將資料打散, 因為所處理的資料是有順序性的樣本資料 (負評在前, 然後才是正評)
np.random.shuffle(indices)                                 # 同上
data = data[indices]                                       # 同上
labels = labels[indices]                                   # 同上

x_train = data[:training_samples]                                         # 切割資料
y_train = labels[:training_samples]                                       # 同上
x_val = data[training_samples: training_samples + validation_samples]     # 同上
y_val = labels[training_samples: training_samples + validation_samples]   # 同上

#=====================================================================#
glove_dir = os.path.join(os.getcwd(), 'glove.6B')

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='UTF-8')
for line in f:
    values = line.split()                                  # 切割
    word = values[0]                                       # 取第一個作為該單字代表index，字典的key
    coefs = np.asarray(values[1:], dtype='float32')        # 用np陣列儲存代表單字的多維向量資料
    embeddings_index[word] = coefs                         # 存入字典索引
f.close()

print('共有 %s 個文字嵌入向量' % len(embeddings_index))


embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():                         # 讀取各個我們這裡有的字
    if i < max_words:                                      # 範圍限定
        embedding_vector = embeddings_index.get(word)      # 從上面引入的多維代表向量中配對我們這裡也有的
        if embedding_vector is not None:                   # 如果上面有拿到該字的代表多維向量的話
            embedding_matrix[i] = embedding_vector         # 將可配對的重新編號後存入。嵌入向量索引中找不到的文字將沒有向量 為 0

#=====================================================================#

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=maxlen))       # (樣本數, 嵌入向量維度, ###)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()



model.layers[0].set_weights([embedding_matrix])            # 嵌入向量
model.layers[0].trainable = False                          # 凍結嵌入層



model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save('11_model.h5')
model.save_weights('11_pre_trained_glove_model.h5')



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('11_Maccuracy.png')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('11_Mloss.png')

plt.show()
