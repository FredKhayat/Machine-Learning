import Neurons2 as Neurons
import matplotlib.pyplot as plt
import random

# Training data
x_train = [[1,1],[1,0],[0,1],[0,0]]
y_train = [[0],[1],[1],[0]]


# Network
i1 = Neurons.Input()
i2 = Neurons.Input()

a1 = Neurons.Sigmoid([i1, i2])
a2 = Neurons.Sigmoid([i1, i2])

b1 = Neurons.Sigmoid([a1, a2])

o1 = Neurons.Sigmoid([b1])

i1.set_value(2)
i2.set_value(5)

# Backprop
for i in range(10000):
    rand = random.randint(0,len(y_train) - 1)
    i1.set_value(x_train[rand][0])
    i2.set_value(x_train[rand][1])
    
    o1.quadratic_cost(y_train[rand][0])
        
# Test
resolution = 200
plot = [[] for i in range(resolution)]
for i in range (resolution):
    for j in range (resolution):
        i1.set_value(i / (resolution - 1))
        i2.set_value(j / (resolution - 1))
        plot[i].append(o1.value)
        
plt.imshow(plot, cmap =plt.cm.Blues)
