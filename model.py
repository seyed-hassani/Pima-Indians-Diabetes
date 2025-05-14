import numpy as np
class Model:
    def __init__(self):
        self.w1 = np.random.randn(1000,9) # TODO: random matrix with size (1000, number_of_input_feautres)
        self.w2 = np.random.randn(1,1000)# TODO: random matrix with size (1, 1000)

    def predict(self, inputs):
        x = inputs

        Z_1 = self.w1.dot(x) # TODO
        A_1 = np.maximum(0, Z_1)  # TODO

        Z_2 = self.w2.dot(A_1) # TODO
        A_2 = 1 / (1 + np.exp(-Z_2)) # TODO

        return A_1, A_2

    def update_weights_for_one_epoch(self, inputs, outputs, learning_rate):
        x, y_true = inputs, outputs
        A_1, A_2 = self.predict(inputs)

        n = x.shape[1] # TODO (n = number of samples)

        shared_coefficient = (2 * learning_rate / n)* (y_true.reshape(1, -1)
                                                        - A_2) * A_2 * (1 - A_2)
        relu_gradient = np.where(A_1 > 0, 1, 0)
        
        self.w1 = self.w1 + (self.w2.T @ shared_coefficient) * relu_gradient @ x.T
        self.w2 = self.w2 + shared_coefficient @ A_1.T # TODO

    def fit(self, inputs, outputs, learning_rate, epochs):
        for i in range(epochs):
            self.update_weights_for_one_epoch(inputs, outputs, learning_rate)

