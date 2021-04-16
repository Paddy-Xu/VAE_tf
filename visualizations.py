import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class visualize():
    def __init__(self, row=3, col=3):
        self.row = row
        self.col = col
        self.gt = None
        self.pred = None
        self.index = None

    def setTrue(self, gt):
        self.gt = gt

    def setPred(self, pred):
        self.pred = pred

    def setIndex(self):
        n = self.pred.shape[0] if self.pred is not None else self.gt.shape[0]
        self.index = np.random.randint(low=0, high=n, size=self.row * self.col)

    def vis(self, preds=False):
        fig, axs = plt.subplots(nrows=self.row, ncols=self.col, constrained_layout=False)
        img = self.pred[self.index] if preds else self.gt[self.index]
        for r in range(self.row):
            for c in range(self.col):
                axs[r, c].imshow(tf.reshape(img[r + c], shape=(28, 28)).numpy().astype("float32"),
                                 cmap='gray')
        plt.show()


# import tensorflow_probability as tfp


def plot_latent_images(model, n, digit_size=28):
    """Plots n x n digit images decoded from the latent space."""

    norm = tfp.distributions.Normal(0, 1)
    # norm = tfp.distributions.Normal(0, 1)

    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.decoder(z)
            x_decoded = tf.keras.layers.Activation('sigmoid')(x_decoded)

            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
