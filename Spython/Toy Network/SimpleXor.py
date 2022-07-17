import Neurons
import random
import matplotlib.pyplot as plt
from contextlib import suppress

# Training data
x_train = [[1,1],[1,0],[0,1],[0,0],[0.5,0],[0,0.5],[1,0.5],[0.5,1]]
y_train = [[0],[0],[0],[0],[1],[1],[1],[1]]


# Network
i1 = Neurons.Input()
i2 = Neurons.Input()

a1 = Neurons.Sigmoid([i1, i2])
a2 = Neurons.Sigmoid([i1, i2])
a3 = Neurons.Sigmoid([i1, i2])
a4 = Neurons.Sigmoid([i1, i2])

b1 = Neurons.Sigmoid([a1, a2, a3, a4])
b2 = Neurons.Sigmoid([a1, a2, a3, a4])

o1 = Neurons.Sigmoid([b1, b2])

w1 = Neurons.Watch([a1, a2, a3, a4])
w2 = Neurons.Watch([b1, b2], w1)
w3 = Neurons.Watch([o1], w2)



def backpropagate(iterations):
    # Backpropagation
    for i in range(iterations):
        rand = random.randint(0,len(y_train) - 1)
        i1.value = x_train[rand][0]
        i2.value = x_train[rand][1]
        w1.activate()
        
        o1.quadratic_cost(y_train[rand][0])
        
        w3.backpropagate(10)
    


def test_network():
    # Plot color map
    resolution = 200
    plot = [[] for i in range(resolution)]
    for i in range (resolution):
        for j in range (resolution):
            i1.value = i / (resolution - 1)
            i2.value = j / (resolution - 1)
            w1.activate()
            plot[i].append(o1.value)
        
        
    # Plot scatterplots and plots
    fig, axs = plt.subplots(3,2, figsize = (16,12))
    axs[0, 0].imshow(plot, cmap=plt.cm.Blues)
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Color Map')
    
    x_axis = [x for x in range(len(o1.deltas_history))]
    
    axs[1, 0].scatter(x_axis, o1.deltas_history)
    axs[0, 0].axis('off')
    axs[1, 0].axes.get_xaxis().set_visible(False)
    axs[2, 0].axes.get_xaxis().set_visible(False)
    axs[1, 0].set_title('B Deltas')
    axs[2, 0].set_title('A Deltas')
    for n in w2.neurons:
        axs[1,0].scatter(x_axis, n.deltas_history)
    for n in w1.neurons:
        axs[2,0].scatter(x_axis, n.deltas_history)
    
    axs[0, 1].axes.get_xaxis().set_visible(False)
    axs[1, 1].axes.get_xaxis().set_visible(False)
    axs[2, 1].axes.get_xaxis().set_visible(False)
    axs[0, 1].set_title('O1 Weights')
    axs[1, 1].set_title('B Weights')
    axs[2, 1].set_title('A Weights')
    axs[0, 1].plot(o1.weights_history)
    for n in w2.neurons:
        axs[1,1].plot(n.weights_history)
    for n in w1.neurons:
        axs[2,1].plot(n.weights_history)

    plt.show()


answer = 1000
while (answer != 0):
    backpropagate(answer)
    test_network()
    
    answer = int(input("Backpropagation iterations: "))

    



    