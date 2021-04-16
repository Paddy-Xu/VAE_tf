import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import mnist
from spektral.layers import GCNConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils import normalized_laplacian
from spektral.utils import normalized_laplacian, normalized_adjacency
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data import MixedLoader
from spektral.datasets.mnist import MNIST
from spektral.layers import GCNConv, GlobalSumPool
from spektral.layers.ops import sp_matrix_to_sp_tensor

learning_rate = 1e-3  # Learning rate for SGD
es_patience = 200     # Patience fot early stopping

# Parameters
batch_size = 32  # Batch size
epochs = 1000  # Number of training epochs
patience = 10  # Patience for early stopping
l2_reg = 5e-4  # Regularization rate for l2


# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(10, activation="softmax")  # MNIST has 10 classes

    def call(self, inputs):
        x, a = inputs

        #a = normalized_laplacian(a)

        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output



# Build model
class model2(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv32 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.conv64= GCNConv(64, activation="elu", kernel_regularizer=l2(l2_reg))
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(10, activation="softmax")  # MNIST has 10 classes

    def call(self, inputs):
        x, a = inputs

        #a = normalized_laplacian(a)

        x = self.conv32([x, a])
        x = self.conv64([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

'''
def model2(adj, N=28*28, F=1):
    @tf.function
    def getInput(A_in):
        res = Input(tensor=A_in,
                     name='LaplacianAdjacencyMatrix')
        return res
    n_out = 10  # Dimension of the target
    X_in = Input(shape=(N, F))
    # #norm_adj = normalized_adjacency(adj)

    # Pass A as a fixed tensor, otherwise Keras will complain about inputs of
    # different rank.
    #
    # fltr = normalized_laplacian(adj)
    # A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr),
    #              name='LaplacianAdjacencyMatrix')

    # A_in = Input(tensor=getInput(),
    #              name='LaplacianAdjacencyMatrix')
    A_in = getInput(adj)
    graph_conv_1 = GCNConv(32,
                           activation='elu',
                           kernel_regularizer=l2(l2_reg),
                           use_bias=True)([X_in, A_in])

    graph_conv_2 = GCNConv(64,
                           activation='elu',
                           kernel_regularizer=l2(l2_reg),
                           use_bias=True)([graph_conv_1, A_in])

    flatten = Flatten()(graph_conv_2)
    fc = Dense(512, activation='relu')(flatten)
    output = Dense(n_out, activation='softmax')(fc)

    # Build model
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model
'''