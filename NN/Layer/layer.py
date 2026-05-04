from __future__ import annotations
import numpy
from NN.Activation.activation import sigmoid



class Layer:

    def __init__(self, input_dim: int = 1, output_dim: int = 1, activation_function = sigmoid, previous_layer: Layer | None = None, next_layer: Layer | None = None):

        self.input_dim = input_dim # dimension (number of Neurons) of previous Layer
        self.output_dim = output_dim # dimension (number of Neurons) of current Layer

        #self.weights = numpy.random.rand(output_dim, input_dim + 1) if not weights else weights # include bias
        self.weights = numpy.array([-3, -3, 2]).reshape((1, -1))
        self.activation_function = activation_function # activation function of layer

        self.next_layer = next_layer
        self.prev_layer = previous_layer

    def get_next_layer(self):
        return self.next_layer

    def get_prev_layer(self):
        return self.prev_layer

    def forward(self, x_input : numpy.ndarray):
        """Calculate the forward pass of the Layer"""

        x = numpy.append(1, x_input)

        assert self.weights.shape[1] ==  x.shape[0], "Weights and Feature input does not have the same dimension!"
        z = self.weights @ x # preactivation
        a = self.activation_function(z) # with activation function

        return a # shape (self.output_dim, 1)

    def backward(self):

        pass











