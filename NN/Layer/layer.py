from __future__ import annotations
import numpy
from NN.Activation.activation import Sigmoid



class Layer:

    def __init__(self, input_dim: int = 1, output_dim: int = 1, activation_function = Sigmoid):

        self.input_dim = input_dim # dimension (number of Neurons) of previous Layer
        self.output_dim = output_dim # dimension (number of Neurons) of current Layer

        #self.weights = numpy.random.rand(output_dim, input_dim + 1) if not weights else weights # include bias
        self.weights = numpy.array([-3, -3, 2]).reshape((1, -1))
        self.activation_function = activation_function # activation function of layer

        self._cache = None

    def forward(self, x_input : numpy.ndarray):
        """Calculate the forward pass of the Layer"""
        x = numpy.append(1, x_input)

        assert self.weights.shape[1] ==  x.shape[0], "Weights and Feature input does not have the same dimension!"
        z = self.weights @ x # preactivation
        a = self.activation_function.forward(z) # call the forward pass in the activation
        self._cache = (x, z) # store the output
        return a # shape (self.output_dim, 1)

    def backward(self, dout: float):

        x, z = self._cache # get the inputs of the previous layer

        da = self.activation_function.backward(z) # derivative of activation layer
        dw = numpy.transpose(x) * da * dout # chain rule (dout =: upstream gradient)

        return dw











