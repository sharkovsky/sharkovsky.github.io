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

weights0 = np.load('weights_0.npy')
biases0 = np.load('biases_0.npy')
weights1 = np.load('weights_1.npy')
biases1 = np.load('biases_1.npy')
biases2 = np.load('biases_2.npy')

input_shape = weights0.shape[1]
hidden_shape0 = weights0.shape[0]
hidden_shape1 = weights1.shape[0]

rbm = RBM.RBM(input_shape)
rbm.add_layer( hidden_shape0 )
rbm.add_layer( hidden_shape1 )

rbm.weights[0] = weights0
rbm.biases[0] = biases0
rbm.weights[1] = weights1
rbm.biases[1] = biases1
rbm.biases[2] = biases2

while True:
    print('input a sample number to analyze')
    selection = input()
    if not selection:
        break
    idx = int(selection)
    if idx > X.shape[0]:
        print('Error: maximum input value is', X.shape[0] )
        continue
    plt.figure()
    plt.subplot(3,1,1)
    X_in = X[idx,:].reshape(28,28)
    plt.imshow(X_in, cmap='gray')
    plt.colorbar()
    plt.subplot(3,1,2)
    X_in = X[idx,:].reshape(1,784)
    out = rbm.positive_phase_inference( X_in )
    plt.imshow(rbm.layers[0].h.reshape((28,28)), cmap='gray')
    plt.colorbar()
    print(out)
    plt.subplot(3,1,3)
    rbm.dream(out.T)
    plt.imshow(rbm.layers[0].h.reshape((28,28)), cmap='gray')
    plt.colorbar()
    plt.show()

