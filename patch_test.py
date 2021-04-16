import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 60000 * 28 * 28

x_train = np.expand_dims(x_train, -1)
x_train = tf.convert_to_tensor(x_train)

patches = tf.image.extract_patches(images=x_train[:10],
                         sizes=[1, 14, 14, 1],
                         strides=[1, 14, 14, 1],
                         rates=[1, 1, 1, 1],
                         padding='VALID')


fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=False)

for imgs in patches:
    count = 0
    for r in range(2):
        for c in range(2):
            axs[r, c].imshow(tf.reshape(imgs[r,c],shape=(14,14)).numpy().astype("float32"))
            count += 1
plt.show()

