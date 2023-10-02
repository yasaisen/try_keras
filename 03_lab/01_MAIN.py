from keras.preprocessing.text import Tokenizer  # 匯入 Keras 分詞器

samples = ['The cat sat on the mat.', 'The dog ate my homework.']  # 初始資料

tokenizer = Tokenizer(num_words=1000) # 1. 建立一個分詞器, 設定上僅考慮 1000 個最常用的文字 (token), 也就是只會看初始資料的前 1000 個文字
tokenizer.fit_on_texts(samples)  # 建立文字對應的索引值, 一樣依出現順序來決定, 0 一樣不使用

print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(samples)  # 將初始資料中的文字轉換成對應的索引值 list
print(sequences) # [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
#不區分大小寫

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 可以直接取得 one-hot 的二進位表示方式。此分詞 tokenizer 支援除了 one-hot 編碼以外, 也有支援其他的向量化方法
print(one_hot_results.shape) # (2, 1000) 共 2 個樣本, 每個樣本中的文字對應到的 token 位置 (1000個)

print(one_hot_results[0][:15])
print(one_hot_results[1][:15])

word_index = tokenizer.word_index  # 計算完成後, 取得文字與索引間的對應關聯 
print(word_index) # {'the': 1, 'cat': 2, 'sat': 3, ... 'my': 8, 'homework': 9}
print('找到 %s 個唯一的 tokens.' % len(word_index)) 