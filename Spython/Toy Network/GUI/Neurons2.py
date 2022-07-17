import random
import math

# =============================================================================
# Sigmoid Neuron
# =============================================================================
class Sigmoid():
    def __init__(self, neurons = []):
        self.weights, self.bias = [], random.gauss(0,1)
        self.next_neurons, self.neurons = [], []    
        self.value, self.delta = 0,0
        self.learning_rate = 4
        
        self.signal_count = len(neurons)
        self.backprop_signal_count = 0
        self.deltas_history, self.weights_history = [0], []
        
        for n in neurons:
            self.connect(n)
        
    
    def activate(self):
        self.value = 0
        for n,w in zip(self.neurons, self.weights):
            self.value += n.value * w
        self.value = self.sigmoid(self.value + self.bias)
        
        
    def backpropagate(self):
        self.delta = self.delta * self.d_sigmoid(self.value)
        self.deltas_history.append(self.delta)
        self.weights_history.append(self.weights.copy())
            
        
        self.bias -= self.delta * self.learning_rate
        for idx, n in enumerate(self.neurons):
            n.delta += self.delta * self.weights[idx]
            self.weights[idx] -= self.delta * n.value * self.learning_rate
            
        self.delta = 0
    
    
    def signal(self):
        self.signal_count -= 1
        if self.signal_count == 0:
            self.activate()
            self.signal_count = len(self.neurons)
            for n in self.next_neurons:
                n.signal()
                
    def backprop_signal(self):
        self.backprop_signal_count -= 1
        if self.backprop_signal_count == 0 or len(self.next_neurons) == 0:
            self.backpropagate()
            self.backprop_signal_count = len(self.next_neurons)
            for n in self.neurons:
                n.backprop_signal()
    
    
    def connect(self, neuron):
        self.neurons.append(neuron)
        self.weights.append(random.gauss(0,1))
        neuron.next_neurons.append(self)
        self.signal_count += 1
        if (type(neuron) != Input):
            neuron.backprop_signal_count += 1
            
    def disconnect(self, neuron):
        i = self.neurons.index(neuron)
        self.neurons.pop(i)
        self.weights.pop(i)
        neuron.next_neurons.remove(self)
        self.signal_count -= 1
        if (type(neuron) != Input):
            neuron.backprop_signal_count -= 1
    
    def quadratic_cost(self, target):
        self.delta = self.value - target
        self.backprop_signal()
    
    def reset(self, previous_neuron = False):
        if not previous_neuron:
            self.weights = [random.gauss(0,1) for i in self.weights]
            self.bias = random.gauss(0, 1)
        else:
            self.weights[self.neurons.index(previous_neuron)] = random.gauss(0, 1)
            
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def d_sigmoid(self, x):
        return x * (1 - x)
            

# =============================================================================
# Input Neuron
# =============================================================================
class Input():
    def __init__(self, v = 0):
        self.value = v
        self.weights = 0
        self.delta = 0
        self.bias = 0
        self.next_neurons = []        
    
    def set_value(self, value):
        self.value = value
        for n in self.next_neurons:
            n.signal()
            
    def backprop_signal(self):
        pass
    