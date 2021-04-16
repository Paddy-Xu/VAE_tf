from convVAE import *
from data_gen import *
from visualizations import *


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

vae = VariationalAutoEncoder(original_dim=784, latent_dim=2)
optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint = tf.train.Checkpoint(model=vae, opt=optimizer)
checkpoint.restore(tf.train.latest_checkpoint('./save'))

#print(vae.summary())

data_gen = mnist_vae_loader(batch_size=128, flatten= False)
train_dataset, test_dataset = data_gen.get_dataset()


test_batch = iter(test_dataset).next()
preds = vae(test_batch)
preds = tf.keras.layers.Activation('sigmoid')(preds)
preds = preds.numpy()
vis = visualize()
vis = visualize()
test_batch = test_batch.numpy()
vis.setTrue(test_batch)
vis.setPred(preds)
vis.setIndex()
vis.vis(preds=True)
vis.vis(preds=False)

plot_latent_images(vae, 20)


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train)