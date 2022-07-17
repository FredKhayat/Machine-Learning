import numpy as np
from NeuralNetwork import NeuralNetwork
from NeuralNetwork2 import NeuralNetwork2
from Correction import Network
import mnistReader
import time

# =============================================================================
# Data Cleaning
# =============================================================================
X_train = np.load("data/X_train.npy")
Y_train = np.load("data/Y_train.npy")
X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

training_data = [(x, y) for x, y in zip(X_train, Y_train)]
test_data = [(x, np.argmax(y)) for x, y in zip(X_test, Y_test)]

X_train = np.squeeze(X_train).T
Y_train = np.squeeze(Y_train).T


# 50 epochs en 120 sec
# =============================================================================
# Training
# =============================================================================
brain = NeuralNetwork2([784,30,10],
                        cost = NeuralNetwork2.quadratic_cost,
                        reg = NeuralNetwork2.none_reg)
start = time.time()
brain.SGD(X_train,Y_train, 20, 5, 0.25, 5, True, test_data)
end = time.time()
print("Time :", end - start)

# brain = NeuralNetwork2.load('good_brain.txt')
# brain.test_network(test_data)


# start = time.time()
# brain = Network([784,30,10])
# brain.SGD(training_data,20,10,0.2,5,test_data,monitor_evaluation_accuracy=True)
# end = time.time()
# print("Time :", end - start)