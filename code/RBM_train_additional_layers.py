import numpy as np
import sys

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home="/home/cremones/svago/machine-learning/data/mnist")

from sklearn.utils import shuffle
mnist.data.shape
X = mnist.data
X = X - 0.5*np.max(X)
X = X/float(np.max(X))

import RBM

weights0 = np.load('old_model_long_training/weights_0.npy')
biases0 = np.load('old_model_long_training/biases_0.npy')
biases1 = np.load('old_model_long_training/biases_1.npy')

input_shape = weights0.shape[1]
hidden_shape0 = weights0.shape[0]

rbm = RBM.RBM(input_shape)
rbm.add_layer( hidden_shape0 )

rbm.weights[0] = weights0
rbm.biases[0] = biases0
rbm.biases[1] = biases1

rbm.add_layer(50)

X_train = X[:60000,:]
n_epochs = 10
for epoch_n in range(n_epochs):
    print('====== epoch ', epoch_n, '=======')
    rbm.train_only_last_layer_with_CD( X_train, 0.005 )

counter = 0
for W in rbm.weights:
    np.save('weights_'+str(counter), W)
    counter += 1

counter = 0
for b in rbm.biases:
    np.save('biases_'+str(counter), b)
    counter += 1
