import numpy as np

X_train = np.load("data/X_train.npy")
Y_train = np.load("data/Y_train.npy")
X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

X_train = np.squeeze(X_train).T
X_test = np.squeeze(X_test).T