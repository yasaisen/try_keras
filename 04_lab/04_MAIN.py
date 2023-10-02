#################   7 - 18
# from keras import layers, Input

# x = Input(batch_shape=(1000, 28, 28, 256))
# print(x.shape)

# branch_a = layers.Conv2D(64, 1, activation='relu', strides=2)(x)
# print(branch_a.shape)

# branch_b = layers.Conv2D(128, 1, activation='relu')(x)
# branch_b = layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(branch_b)
# print(branch_b.shape)

# branch_c = layers.AveragePooling2D(3, strides=2, padding='same')(x)
# branch_c = layers.Conv2D(128, 3, activation='relu', padding='same')(branch_c)
# print(branch_c.shape)

# branch_d = layers.Conv2D(128, 1, activation='relu')(x)
# branch_d = layers.Conv2D(128, 3, activation='relu', padding='same')(branch_d)
# branch_d = layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(branch_d)
# print(branch_d.shape)

# output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
# print(output.shape)


#################   7 - 22


from keras import layers, Input

x = Input(batch_shape=(1000, 32, 32, 256))
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
z = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
print(z.shape)

t = layers.MaxPool2D(2, strides=2)(z)
print(t.shape)

residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
print(residual.shape)

output = layers.add([t, residual])
print(output.shape)












