from keras import backend as K

tf_session = K.get_session()    # 取得後端 tensorflow 的 session
tensor = K.constant([[1, -2], [3, 4]])  # 以後端建立一個張量常數

# 1. shape() 取得張量的 shape 大小
print('shape:')
print(K.shape(tensor).
      eval(session=tf_session))    # 張量 shape：[2 2]

# 2. cast() 將張量元素轉成浮點數型態
print('cast:')
print(K.cast(tensor, 'float32').    # 張量浮點數：[[-1.  2.]
      eval(session=tf_session))     #             [ 3.  4.]]

# 3. abs() 將張量元素取絕對值 
print('abs:')                                                     
print(K.abs(tensor).                # 張量絕對值 [[1 2]
      eval(session=tf_session))     #            [3 4]]
                                
# 4. prod() 將張量元素相乘 
print('prod:')                                                   
print(K.prod(tensor).
      eval(session=tf_session))    # 張量元素相乘：-24

# 5. sum() 將張量元素相加
print('sum:') 
print(K.sum(tensor).
      eval(session=tf_session))    # 張量元素相加：6

# 6. maximum() 將小於門檻值的張量元素以門檻值取代
print('maximum:') 
print(K.maximum(tensor, 2).
      eval(session=tf_session))       # 比 2 小的元素被 2 取代了

# 7. square() 將張量元素進行平方運算
print('square:') 
print(K.square(tensor).             # 張量平方值 [[  1.   4.]
      eval(session=tf_session))     #             [  9.  16.]]

# 8. permute_dimensions() 將 shape 依據變換規則進行變換
    # shape (1, 2, 3) --> (3, 1, 2)  因為指定了 (2,0,1) 代表:位置 2 變成 位置 0
    #                                                       位置 0 變成 位置 1    
    #                                                       位置 1 變成 位置 2
print('permute_dimensions:')    
tensor = K.constant([[[1, 2, 3], 
                       [4, 5, 6]]])  # 以後端建立一個 3 維張量常數
print(K.permute_dimensions(tensor, (2, 0, 1)).            
      eval(session=tf_session))  

# 9. batch_flatten() 將張量拉平成矩陣，即 shape = (1, n)
print('batch_flatten:') 
tensor = K.constant([[[1, 2], 
                       [3, 4],
                       [5, 6]]])  
print(K.batch_flatten(tensor).            
      eval(session=tf_session))  

# 10. transpose() 將 shape 倒轉, 例如 (1,2,3) -> (3,2,1) 
print('transpose:') 
tensor = K.constant([[[1, 2, 3], 
                       [4, 5, 6]]])  # 以後端建立一個 3 維張量常數 
print(K.transpose(tensor).            
      eval(session=tf_session)) 

# 11. dot() 將進行矩陣內積
print('dot:') 
matrixA = K.constant([[1, 3, 5]])           # 建立矩陣 A, shape = (1,3)
matrixB = K.constant([[[2],[4],[6]]])       # 建立矩陣 B, shape = (3,1)
print(matrixB.eval(session=tf_session))
print(K.dot(matrixA, matrixB).            
      eval(session=tf_session))      # A 與 B 內積 = [[[1x2 + 3x4 + 5x6]]] = [[[44.]]]
                                     # 註：若 B 為 A 的轉置矩陣, 則內積結果稱為格拉姆矩陣


# 12. pow() 對張量元素進行特定次方運算
tensor = K.constant([[1, -2], [3, 4]])  
print('pow:') 
print(K.pow(tensor, 2).            # [[ 1.  4.]
      eval(session=tf_session))     #  [ 9. 16.]]