import numpy as np

def reweight_distribution(original_distribution, temperature=2):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)



ori_dstri = np.array([0.8, 0.1, 0.1])   # a、b、c 的機率分布

new_dstri = reweight_distribution(ori_dstri, 
                                  temperature=0.01)  # 使用溫度 0.01 
print(new_dstri)    # [1.00000000e+00 4.90909347e-91 4.90909347e-91]

new_dstri = reweight_distribution(ori_dstri, 
                                  temperature=2) # 使用預設溫度 2
print(new_dstri)    # [0.58578644 0.20710678 0.20710678]

new_dstri = reweight_distribution(ori_dstri, 
                                  temperature=10) # 使用溫度 10
print(new_dstri)    # [0.38102426 0.30948787 0.30948787]




