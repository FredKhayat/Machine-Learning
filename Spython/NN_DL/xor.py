import numpy as np
import matplotlib.pyplot as plt
from SimpleNN import SimpleNN
from NeuralNetwork import NeuralNetwork
from SimpleNN2 import SimpleNN2
from NeuralNetwork2 import NeuralNetwork2

brain = NeuralNetwork(2,4,1)

trainingSet = np.array([[0,1],
                        [1,0],
                        [1,1],
                        [0,0]])

targetSet = np.array([1,1,0,0])

print(brain.feed_forward(trainingSet[0].reshape(2,1)))
print(brain.feed_forward(trainingSet[1].reshape(2,1)))
print(brain.feed_forward(trainingSet[2].reshape(2,1)))
print(brain.feed_forward(trainingSet[3].reshape(2,1)))
print()

#Train the neural network
for i in range(10000):
    r = np.random.randint(4)
    inputs = trainingSet[r].reshape(2,1)
    target = targetSet[r].reshape(1,1)
    brain.train(inputs,target, 2)


print(brain.feed_forward(trainingSet[0].reshape(2,1)))
print(brain.feed_forward(trainingSet[1].reshape(2,1)))
print(brain.feed_forward(trainingSet[2].reshape(2,1)))
print(brain.feed_forward(trainingSet[3].reshape(2,1)))


# =============================================================================
# PYPLOT
# =============================================================================
resolution = 200
figSize = 2

pixels = np.empty(shape = (resolution, resolution))
plt.figure(figsize = (figSize,figSize), dpi = resolution/figSize)
plt.axis('off')

for i in range(resolution):
    for j in range(resolution):
        inputs = np.array([[i/resolution],[j/resolution]])
        pixels[i][j] = brain.feed_forward(inputs)
        
cmap = plt.imshow(pixels)
cmap.set_cmap("Blues")

        
        

