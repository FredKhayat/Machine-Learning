import numpy as np
import json

class NeuralNetwork2:
    quadratic_cost = 'quadratic'
    cross_entropy_cost = 'cross_entropy'
    L2_reg = 'L2'
    L1_reg = 'L1'
    none_reg = 'none'    
    
    
    def __init__(self, size , cost = 'quadratic', reg = 'none'):
        self.layerNumber = len(size) - 1
        
        self.init_cost(cost)
        self.init_reg(reg)
        self.training_size = 0
        
        # Initialize weights and biases
        self.biases = [np.random.randn(x, 1) for x in size[1:]]
        self.weights = [np.random.randn(x, y)/np.sqrt(y)for x, y in zip(size[1:], size[:-1])]
            
        
    def feed_forward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(w, x) + b)
        return x
    
    
    def SGD(self, x, y, batch_size = 1, epochs = 1, learning_rate = 1, reg_rate = 1,
            early_stop = False, test_data = None):
        
        self.training_size = x.shape[1]
        
        def train_epoch(x, y, epoch):
            p = np.random.permutation(x.shape[1])
            x = x[:, p]
            y = y[:, p]
            for k in range(0, x.shape[1], batch_size):
                self.backprop(x[:, k:k + batch_size], y[:,k:k + batch_size], batch_size, learning_rate, reg_rate)
            
            if test_data != None:
                return self.test_network(test_data,epoch)
            else:
                print(f"Epoch {epoch} complete!")
                return 0
        
        
        if(early_stop) and test_data != None:
            best_score, counter, current_epoch = 0, 0, 0
            best_weights, best_biases, initial_lr = 0, 0, learning_rate
            while learning_rate > initial_lr/128:
                counter, current_epoch = counter + 1, current_epoch + 1
                current_score = train_epoch(x, y, current_epoch)
                if best_score < current_score:
                    best_weights, best_biases = self.weights, self.biases
                    best_score, counter = current_score, 0
                if counter > epochs:
                    learning_rate *= 0.5
                    
            self.weights, self.biases = best_weights, best_biases
        else:
            for epoch in range(epochs):
                train_epoch(x, y, epoch + 1)
                
                
                
# =============================================================================
#     Auxiliary Methods
# =============================================================================
    def backprop(self, x, y, batch_size, learning_rate, reg_rate):
        outputs = []
        activations = [x]
        for i in range(self.layerNumber):
            outputs.append(np.dot(self.weights[i], activations[i]) + self.biases[i])
            activations.append(self.sigmoid(outputs[i]))
            
        # Back Propagation
        error = self.cost(activations[-1], y)
        delta = error * self.d_sigmoid(outputs[-1])
        
        delta_w = [0 for w in self.weights]
        delta_b = [0 for b in self.biases]
        delta_w[-1] = np.dot(delta, activations[-2].T)
        delta_b[-1] = np.sum(delta, axis = 1, keepdims = True)
        
        for i in range(2, self.layerNumber + 1):
            delta = np.dot(self.weights[-i + 1].T, delta) * self.d_sigmoid(outputs[-i])
            delta_b[-i] = np.sum(delta, axis=1, keepdims=True)
            delta_w[-i] = np.dot(delta, activations[-i - 1].T)
            
        self.weights = [x - y * (learning_rate/batch_size) - self.reg(x) * reg_rate * learning_rate for x, y in zip(self.weights, delta_w)]
        self.biases = [x - y * (learning_rate/batch_size) for x, y in zip(self.biases, delta_b)]
        
    
    def test_network(self, test_data, epoch = None):
        correct = 0
        
        for d in test_data:
            if np.argmax(self.feed_forward(d[0])) == d[1]:
                correct += 1
            
        if epoch != None:
            print (f"Epoch {epoch}: {(correct/ len(test_data)) * 100}%")
        else:
            print(f"{(correct/ len(test_data)) * 100}% correct guesses!")
            
        return correct/len(test_data)               
    
    
# =============================================================================
#   Initialization Methods
# =============================================================================
    def init_cost(self, cost):
        self.cost_name = cost
        if cost == self.quadratic_cost:
            self.cost = lambda output, target: output - target
        
        if cost == self.cross_entropy_cost:
            self.cost = lambda output, target: (1-target)/(1-output) - target/output
            
    def init_reg(self, reg):
        self.reg_name = reg
        if reg == self.none_reg:
            self.reg = lambda w: 0
        if reg == self.L2_reg:
            self.reg = lambda w: w/self.training_size
        if reg == self.L1_reg:
            self.reg = lambda w: np.sign(w)/self.training_size
            
            
# =============================================================================
#   Activation Functions        
# =============================================================================
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        
    def relu(self,x):
        temp = x
        temp[temp < 0] = 0
        return temp
    
    def d_relu(self,x):
        temp = x
        temp[temp < 0] = 0
        temp[temp > 0] = 1
        return temp
    
    
# =============================================================================
#   Save and load  
# =============================================================================
    def save(self, filename):
        data = {'size': [len(x) for x in self.biases],
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'cost': self.cost_name,
                'reg': self.reg_name}
        
        with open(filename, 'w') as f:
            json.dump(data, f)
            
    def load(filename):
        with open(filename) as f:
            data = json.load(f)
            
        net = NeuralNetwork2(data['size'], data['cost'], data['reg'])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
    
    
    