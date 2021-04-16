import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

class mnist_vae_loader():
    def __init__(self,batch_size=64, flatten=False):
        self.batch_size = batch_size
        self.flatten = flatten
    def get_gen(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0],-1)
        else:
            x = np.expand_dims(x, -1)
        x = tf.convert_to_tensor(x)

        x = tf.data.Dataset.from_tensor_slices(x)
        #auto = tf.data.experimental.AUTOTUNE
        #x = x.shuffle(buffer_size=auto if auto > 1 else 64).batch(self.batch_size)
        x = x.shuffle(128).batch(self.batch_size)

        return x
    def get_dataset(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        train_dataset = self.get_gen(x_train,)
        test_dataset = self.get_gen(x_test)

        return train_dataset,test_dataset


class mnist_vae_loader_w_gradient():
    def __init__(self,batch_size=64, flatten=False):
        self.batch_size = batch_size
        self.flatten = flatten
    def get_gen(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0],-1)
        else:
            x = np.expand_dims(x, -1)

        x = tf.convert_to_tensor(x)
        def get_grad(x):
            dx, dy = tf.image.image_gradients(tf.expand_dims(x, axis=0))
            gradient = dx+dy
            return gradient[0]

        x = tf.data.Dataset.from_tensor_slices((x))
        grad = x.map(get_grad)

        inputs = tf.data.Dataset.zip((x,grad))

        #gradient = tf.data.Dataset.from_tensor_slices((gradient))

        res = tf.data.Dataset.zip((inputs, x))

        #x = tf.data.Dataset.zip(((x, gradient),x))


        # x = tf.data.Dataset.from_tensor_slices((x,gradient))
        #auto = tf.data.experimental.AUTOTUNE
        #x = x.shuffle(buffer_size=auto if auto > 1 else 64).batch(self.batch_size)
        res = res.shuffle(128).batch(self.batch_size)

        return res

    def get_dataset(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        train_dataset = self.get_gen(x_train)
        test_dataset = self.get_gen(x_test)

        return train_dataset,test_dataset

class mnist_vae_loader_w_gradient_iterator():
    def __init__(self,batch_size=64, epochs=10, flatten=False):
        self.batch_size = batch_size
        self.epochs = epochs
    def get_gen(self, x):
        x = np.expand_dims(x, -1)
        def gen_series():
            while True:
                i = np.random.randint(0, 10)
                yield [x[i],x[i]], x[i]

        res = tf.data.Dataset.from_generator(
            gen_series,
            output_types=(tf.float64, tf.float64),
            #output_shapes=((28,28,1,), (28,28,1,))
        )

        dataset = res.repeat(self.epochs)
        dataset = dataset.batch(self.batch_size)
        #dataset = dataset.map(self.fixup_shape)
        return dataset

    def fixup_shape(self, images, labels):
        images.set_shape([None, 2, 28, 28, 1])
        labels.set_shape([None, 28, 28, 1])  # I have 19 classes
        return images, labels

    def get_dataset(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        train_dataset = self.get_gen(x_train)
        test_dataset = self.get_gen(x_test)

        return train_dataset,test_dataset

# fig, axs = plt.subplots(nrows=3, ncols=3, constrained_layout=False)
# preds = model(1)
#
# for r in range(3):
#     for c in range(3):
#         img = np.random.choice(a=preds, size=1).item()
#         axs[r, c].imshow(tf.reshape(img,shape=(28,28)).numpy().astype("float32"))
# plt.show()
