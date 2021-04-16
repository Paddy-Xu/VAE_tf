
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
import matplotlib.pyplot as plt
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

class data_loader():
      def __int__(self):
            self.data = MNIST()

      def getdata(self):
            # Load data
            self.data
            adj = self.data.a

            # The adjacency matrix is stored as an attribute of the dataset.
            # Create filter for GCN and convert to sparse tensor.

            self.data.a = GCNConv.preprocess(self.data.a)
            self.data.a = sp_matrix_to_sp_tensor(self.data.a)

            # Train/valid/test split
            data_tr, data_te = self.data[:-10000], self.data[-10000:]
            np.random.shuffle(data_tr)
            data_tr, data_va = data_tr[:-10000], data_tr[-10000:]

            # We use a MixedLoader since the dataset is in mixed mode
            loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
            loader_va = MixedLoader(data_va, batch_size=batch_size)
            loader_te = MixedLoader(data_te, batch_size=batch_size)
            return adj, loader_tr, loader_va, loader_te

      def get_one_batch_x(self, batch_size=3):
            loader = MixedLoader(batch_size=batch_size)
            inputs, target = loader.__next__()
            x = inputs[0]
            return x

      def get_adj(self):
            data = MNIST()
            return data.a
# print(adj.shape, 'adjacency matrix')
# plt.matshow(adj.todense())