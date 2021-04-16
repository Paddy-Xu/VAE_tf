from GCN.model import model2, Net
from tensorflow.keras.metrics import sparse_categorical_accuracy
from GCN.data_loader import *
from GCN.train import *
# Create model
#model = Net()
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

class trainer():
    def __init__(self, model=None):
        self.model = model
    def set_model(self, model):
        self.model = model

    # Training function
    def train_custom(self, patience, loader_tr,loader_va,loader_te):

        # Setup training
        best_val_loss = 99999
        current_patience = patience
        step = 0

        # Training loop
        results_tr = []
        for batch in loader_tr:
            step += 1

            # Training step
            inputs, target = batch
            # x, a = inputs
            # a = normalized_laplacian(a)
            # inputs = (x,a)

            loss, acc = self.train_on_batch(inputs, target)
            results_tr.append((loss, acc, len(target)))

            if step == loader_tr.steps_per_epoch:
                results_va = self.evaluate(loader_va)
                if results_va[0] < best_val_loss:
                    best_val_loss = results_va[0]
                    current_patience = patience
                    results_te = self.evaluate(loader_te)
                else:
                    current_patience -= 1
                    if current_patience == 0:
                        print("Early stopping")
                        break

                # Print results
                results_tr = np.array(results_tr)
                results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
                print(
                    "Train loss: {:.4f}, acc: {:.4f} | "
                    "Valid loss: {:.4f}, acc: {:.4f} | "
                    "Test loss: {:.4f}, acc: {:.4f}".format(
                        *results_tr, *results_va, *results_te
                    )
                )

                # Reset epoch
                results_tr = []
                step = 0
    def train_common(self, loader_tr):
        self.model.fit(loader_tr.load(), loader_tr.steps_per_epoch)

    @tf.function
    def train_on_batch(self, inputs, target):
        with tf.GradientTape() as tape:

            predictions = self.model(inputs, training=True)
            loss = loss_fn(target, predictions) + sum(self.model.losses)
            acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, acc


    # Evaluation function
    def evaluate(self, loader):
        step = 0
        results = []
        for batch in loader:
            step += 1
            inputs, target = batch
            predictions = self.model(inputs, training=False)
            loss = loss_fn(target, predictions)
            acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
            results.append((loss, acc, len(target)))  # Keep track of batch size
            if step == loader.steps_per_epoch:
                results = np.array(results)
                return np.average(results[:, :-1], 0, weights=results[:, -1])
