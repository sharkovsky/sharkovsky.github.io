import numpy as np
import sys

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home="/home/cremones/svago/machine-learning/data/mnist")

from sklearn.utils import shuffle
mnist.data.shape
X = mnist.data
X = X - 0.5*np.max(X)
X = X/float(np.max(X))
X = shuffle(X)

import RBM

rbm = RBM.RBM(784)
rbm.add_layer(100)
#rbm.add_layer(50)

X_train = X[:60000,:]
n_epochs = 30
for epoch_n in range(n_epochs):
    print('====== epoch ', epoch_n, '=======')
    rbm.train_with_CD( X_train, 0.005 )

counter = 0
for W in rbm.weights:
    np.save('weights_'+str(counter), W)
    counter += 1


counter = 0
for b in rbm.biases:
    np.save('biases_'+str(counter), b)
    counter += 1
