import Neurons
import Networks
import random
import matplotlib.pyplot as plt
import math
import numpy as np

# Training Data
x_train = [[1,1],[1,0],[0,1],[0,0],[0.5,0],[0,0.5],[1,0.5],[0.5,1]]
y_train = [[0],[0],[0],[0],[1],[1],[1],[1]]

# Backprop 
brain = Networks.SigmoidNetwork([2,4,2,1])
for i in range(10000):
    rand = random.randint(0,len(y_train) - 1)
    brain.backpropagate(x_train[rand], y_train[rand], 5)


# Test Network
resolution = 200
plot = np.zeros((resolution, resolution))
for i in range (resolution):
    for j in range (resolution):

        plot[i,j] = brain.feedforward([i/(resolution-1),j/(resolution-1)])[0]
        
color_map = plt.imshow(plot, cmap=plt.cm.Blues_r)   
    



        
        

    
    