from keras import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
						 #↓1...                   #↓2...
text_input = Input(shape=(None, ), dtype='int32', name='text') 
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input) #← 3...
print(embedded_text.shape)  	#→ (?, ?, 64)
encoded_text = layers.LSTM(32)(embedded_text) #← 4...
print(encoded_text.shape)  #	→ (?, 32)

question_input = Input(shape=(None, ), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input) #5..
print(embedded_question.shape)  	#→ (?, ?, 32)
encoded_question = layers.LSTM(16)(embedded_question)
print(encoded_question.shape)  	#→ (?, 16)
													#↓6...
concatenated = layers.concatenate([encoded_question, encoded_text], axis=-1) 
print(concatenated.shape)  #→ (?, 48)

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated) #← 7...
print(answer.shape)  #→ (?, 500)

model = Model([text_input, question_input], answer) #← 8...
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

#1. shape = (None, ) 代表不限定張量的 shape 大小, 所以文字輸入可以是可變長度的整數序列。
#2. 請注意, 可以選擇是否為輸入命名, 原因為下面程式 7.2 中的訓練方法 2。
#3. 將輸入送進嵌入層, 編碼成大小 64 的文字嵌入向量 (處理 「參考文字」輸入)。
#4. 再透過 LSTM 層將向量序列編碼成單一個向量
#5. 處理「問題」輸入的流程 (與處理「參考文字」輸入的流程相同)
#6. 串接編碼後的「問題」和「參考文字」資料 (向量), 將兩份資料合而為一。axis 參數為 -1 代表以輸入的最後一個軸進行串接。
#7. 最後增加一個 Dense層 (softmax分類器), 將串接向量送入, 輸出模型的結果張量 answer。
#8. 在模型實例化時, 因為有兩個輸入, 所以將它們組成一個 list 一起做為輸入, 而輸出為 answer。


#=====================================================================#


import numpy as np

num_samples = 1000
max_length = 100

# 產生 text 資料：1000 筆, 每筆 100 個字 (數字)
text = np.random.randint(1, text_vocabulary_size, 
                         size=(num_samples, max_length))
#  [  [2, 15, 8000,..... 共 100 個], [],....共 1000 筆  ]  
#      ↑   ↑    ↑         
#     產生 1 ~ 10000 (text_vocabulary_size) 區間的數字 
print(text.shape)       # (1000, 100)

# 產生 question 資料, 與上面 text 產生方式相同
question = np.random.randint(1, question_vocabulary_size, 
                             size=(num_samples, max_length))
print(question.shape)   # (1000, 100)

# 產生 answers 資料, 需為 One-hot 編碼, 共 1000 個正確答案
answers = np.random.randint(0, 1, size=(num_samples, 
                                        answer_vocabulary_size))
#  [  [0, 1, 1,..... 共 100 個], [],.... 共 1000 筆  ]
#      ↑  ↑  ↑         
#     產生 0 ~ 1 的數字 
# 此為分類器要用的 One-encoding 編碼答案    
print(answers.shape)    # (1000, 500)

# 訓練方法 1：使用 list 方式送入資料進行擬合 
#model.fit([text, question], answers, epochs=10, batch_size=128)
# 訓練方法 2：使用 dict 方式送入資料進行擬合, 鍵為 Input 層的名稱, 值為 Numpy 資料
history = model.fit({'text': text, 'question': question}, answers, epochs=10,  batch_size=128)



