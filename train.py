from convVAE import *
from data_gen import *
tf.keras.backend.set_floatx('float64')

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*2-500))])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
class trainer():
    def __int__(self, model, batch_size = 128, optimizer= tf.keras.optimizers.Adam(1e-4)):
        self.optimizer = optimizer
        self.model = model

    def model_type(self, ):
        pass

    def train(sself, set_memory=True, batch_size=128, latent_dim=2):
        if set_memory:
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        vae = VariationalAutoEncoder(original_dim=784, latent_dim=latent_dim)

        optimizer = tf.keras.optimizers.Adam(1e-4)

        vae.compile(optimizer)

        data_gen = mnist_vae_loader(batch_size=batch_size, flatten= False)
        train_dataset, test_dataset = data_gen.get_dataset()

        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # for x in iter(train_dataset).next():
        #     print(x.shape)

        if not os.path.exists('./save'):
            os.makedirs('./save')


        checkpoint = tf.train.Checkpoint(model=vae, opt=optimizer)

        vae.fit(train_dataset, epochs=15)
        checkpoint.save('./save/model.ckpt')

