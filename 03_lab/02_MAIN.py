
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000 #  將文字儲存成大小為 1, 000 的向量。如果有接近 1, 000 個文字(或更多), 將會造成許多雜湊碰撞, 這會降低此編碼方法的準確性
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))[:max_length]:
		index = abs(hash(word)) % dimensionality # ← 將文字雜湊成 0 到 1, 000 之間的隨機整數索引
		results[i, j, index] = 1.
print(results.shape)