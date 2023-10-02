from cv2 import batchDistance
from keras import layers, Input
from keras.models import Model

vocabulary_size = 50000 	#← 文章大小
num_income_groups = 10 	#← 將收入分成 10 群
                            
                          # ↓不限定輸入向量的 shape 大小
posts_input = Input(shape=(None,), dtype='int32', name='posts') 

# 用函數式 API 將輸入向量傳入 Embedding 層, 得到維度 256 的嵌入向量
embedding_posts = layers.Embedding(vocabulary_size, 256)(posts_input)
print(embedding_posts.shape)   # ← (?, ?, 256)

# 以下以函數式 API 將嵌入向量傳入一層層之中進行處理
x = layers.Conv1D(128, 5, activation='relu')(embedding_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)  
x = layers.Dense(128, activation='relu')(x)
print(x.shape)  #← 走過一連串層之後, x.shape 為 (?, 128)

# 接下來將 x 向量分別送入 3 個輸出層。請注意, 
# 需為輸出層指定名稱(原因請見程式 7.5 中的編譯方法 2)

# 預測年紀的輸出層：純量迴歸任務
age_prediction = layers.Dense(1, name='age')(x)

# 預測收入族群的輸出層多分類任務 (10 類)
income_prediction = layers.Dense(num_income_groups, 
                                 activation='softmax', 
                                 name='income')(x)
# 預測性別的輸出層：二元分類任務
gender_prediction = layers.Dense(1, 
                                 activation='sigmoid', 
                                 name='gender')(x)

# 用輸入向量與輸出向量實例化 Model 物件
model = Model(posts_input, 
              [age_prediction, income_prediction, gender_prediction])
                 #↑ 因為輸出向量有 3 個, 所以用 list 來組成

model.summary()


#=====================================================================#


# 編譯方式 1 
model.compile(optimizer='rmsprop', 
              loss=['mse',		#← (需照建立層的順序)
                    'categorical_crossentropy', 
                    'binary_crossentropy'])
# 編譯方式 2 
model.compile(optimizer='rmsprop', 
              loss={'age': 'mse',	#← (需為輸出層指定名稱)
                    'income': 'categorical_crossentropy', 
                    'gender': 'binary_crossentropy'})


#=====================================================================#


# 編譯方式 1 
model.compile(optimizer='rmsprop', 
              loss=['mse',		#← (需照建立層的順序)
                    'categorical_crossentropy', 
                    'binary_crossentropy'],
			  loss_weights=[0.25, 1., 10.])
# 編譯方式 2 
model.compile(optimizer='rmsprop', 
              loss={'age': 'mse',	#← (需為輸出層指定名稱)
                    'income': 'categorical_crossentropy', 
                    'gender': 'binary_crossentropy'},
			  loss_weights={'age': 0.25,
			  			    'income': 1.,
							'gender': 10.})

model.fit(posts,
          [age_targets, income_targets, gender__targets]),
          epochs=20,
          batch_size=64)

model.fit(posts,
          {'age': age_targets,
           'income': income_targets, 
           'gender': gender__targets},
          epochs=20,
          batch_size=64)
