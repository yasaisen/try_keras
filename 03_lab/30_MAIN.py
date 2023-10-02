import os


data_dir = os.path.join(os.getcwd(), 'jena_climate')
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv') # 資料集完整路徑

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]


print(header)  # 
print(len(header))
print(len(lines))


#=====================================================================#


import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
print(float_data.shape)   # 共有 420551 個時間點的天氣資料, 每個包含 14 種天氣數值


#=====================================================================#


from matplotlib import pyplot as plt

temp = float_data[:, 1] # 索引 1 為 temperature 資料
plt.plot(range(len(temp)), temp)

plt.figure()

plt.plot(range(1440), temp[:1440])

plt.show()
