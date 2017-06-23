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

import scipy.ndimage

downsampled_size = 17
downsampled_X = np.zeros( shape=(X.shape[0], downsampled_size*downsampled_size) )
for idx,x in enumerate(X):
    downsampled_X[idx,:] = scipy.ndimage.zoom( x.reshape(28,28), 0.6).reshape(1,downsampled_size*downsampled_size)

# transpose data
X = downsampled_X.T

import BM

bm = BM.BM(1000, downsampled_size*downsampled_size)

X_train = X[:,:100]
n_epochs = 1
for epoch_n in range(n_epochs):
    print('====== epoch ', epoch_n, '=======')
    bm.train_with_CD( X_train, 0.01 )

np.save('BM_W', W)
np.save('BM_b', b)
