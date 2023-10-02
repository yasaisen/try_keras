import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.', 'The mouse is jumping on my bed!']  # 初始資料：每一個樣本是一個輸入項目(在這範例中，樣本是一個句子，但也可以是整個文件)

token_index = {}  # 建立資料中所有 tokens 的索引
for sample in samples:
    for word in sample.split():  # 透過 split()方法對樣本進行分詞。在真實案例中，還要移除樣本中的標點符號與特殊字元
        if word not in token_index:
            token_index[word] = len(token_index) + 1  # 為每個文字指定一個唯一索引。請注意，不要把索引 0 指定給任何文字


max_length = 10  # 將樣本向量化。每次只專注處理每個樣本中的第一個 max_length 文字
#============^^大切

results = np.zeros(shape=(len(samples),  # 用來儲存結果的 Numpy array
                          max_length,
                          max(token_index.values()) + 1))  
print(results.shape) # shape=(2, 10, 11), 共 2 個樣本, 每個樣本只看前 10 個文字, 總樣本共有 10 個 token, 索引號到 11, 因為 0 不用。
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
print(token_index)

print(results)




# import string
# import numpy as np

# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# characters = string.printable  # 所有可印出的 ASCII 字元的字串, '0123456789abc....'
# print(len(characters))

# token_index = dict(zip(characters, range(1, len(characters) + 1)))

# max_length = 50
# results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
# print(results.shape) 

# for i, sample in enumerate(samples):
# 	for j, character in enumerate(sample):
# 		index = token_index.get(character)
# 		results[i, j, index] = 1.
# print(results[0][0])