from GCN.data_loader import *
from GCN.model import *
from GCN.train import *

# Parameters
batch_size = 32  # Batch size
epochs = 1000  # Number of training epochs
patience = 10  # Patience for early stopping
l2_reg = 5e-4  # Regularization rate for l2

data_loader = data_loader()

adj, loader_tr, loader_va, loader_te = data_loader.getdata()

model = Net()
trains = trainer(model)
trains.train_custom(patience, loader_tr, loader_va, loader_te)


model = model2()
trains = trainer(model)

trains.train_custom(patience, loader_tr, loader_va, loader_te)

