import numpy as np

a = np.array([0,2,3,-1,-2,4,-5])

def lol(arr):
    arr[arr<0] = 0
    return arr

lol(a)
print(a)
print(lol(a))




# # =============================================================================
# #   Old Methods
# # =============================================================================
#     def SGD_slow(self, training_data, batch_size = 1, epochs = 1, learning_rate = 1, test_data = None):
#         for epoch in range(epochs):
#             random.shuffle(training_data)
#             batches = [training_data[k : k + batch_size] 
#                        for k in range(0, len(training_data), batch_size)]            
            
#             for batch in batches:
#                 self.update_batch(batch, learning_rate)
                
#             if test_data != None:
#                 self.test_network(test_data,epoch)
#             else:
#                 print("Epoch{epoch + 1} complete!")
                
       
#     def update_batch(self, batch, learning_rate):
#         delta_w = [0 for i in range(self.layerNumber)]
#         delta_b = [0 for i in range(self.layerNumber)]
        
#         for x, y in batch:
#             gradient = self.backprop_slow(x, y)
#             delta_w = [a + b for a, b in zip(gradient[0], delta_w)]
#             delta_b = [a + b for a, b in zip(gradient[1], delta_b)]
            
#         self.weights = [a - b  * (learning_rate/ len(batch)) for a, b in zip(self.weights, delta_w)]
#         self.biases = [a - b * (learning_rate/ len(batch)) for a, b in zip(self.biases, delta_b)]           
       
        
#     def backprop_slow(self, x, y):
#         # Feed Forward
#         outputs = []
#         activations = [x]
#         for i in range(self.layerNumber):
#             outputs.append(np.dot(self.weights[i], activations[i]) + self.biases[i])
#             activations.append(self.sigmoid(outputs[i]))
            
#         # Back Propagation
#         error = activations[-1] - y
#         delta = error * self.d_sigmoid(outputs[-1])
        
#         delta_w = [0 for i in range(self.layerNumber)]
#         delta_b = [0 for i in range(self.layerNumber)]
#         delta_w[-1] = np.dot(delta, activations[-2].T)
#         delta_b[-1] = delta
        
#         for i in range(2, self.layerNumber + 1):
#             delta = np.dot(self.weights[-i + 1].T, delta) * self.d_sigmoid(outputs[-i])
#             delta_w[-i] = np.dot(delta, activations[-i - 1].T)
#             delta_b[-i] = delta
        
#         return (delta_w, delta_b)