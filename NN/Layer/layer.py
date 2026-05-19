from __future__ import annotations
import numpy
from NN.Activation.activation import Sigmoid
from NN.Optimization.optimizer import Optimizer, GradientDescent


class Layer:

    def __init__(self, input_dim: int = 1, output_dim: int = 1, activation_function = Sigmoid):

        self.input_dim = input_dim # dimension (number of Neurons) of previous Layer
        self.output_dim = output_dim # dimension (number of Neurons) of current Layer

        self.weights = None
        self.random_weight_initialization(input_dim=input_dim, output_dim=output_dim)  # include bias

        #self.weights = numpy.array([-3, -3, 2]).reshape((3, 1))
        self.activation_function = activation_function # activation function of layer

        self._cache = None

    def random_weight_initialization(self, input_dim: int, output_dim: int):
            self.weights = numpy.random.rand(input_dim+1, output_dim)
            print(self.weights)

    def forward(self, x_input : numpy.ndarray):
        """Calculate the forward pass of the Layer"""
        print(x_input.shape)

        x = numpy.concatenate((numpy.ones((x_input.shape[0], 1)), x_input), axis=1)

        print(x.shape)
        assert self.weights.shape[0] ==  x.shape[1], "Weights and Feature input does not have the same dimension!"
        z = x @ self.weights # preactivation
        a = self.activation_function.forward(z) # call the forward pass in the activation
        self._cache = (x, z) # store the output
        return a # shape (Batch_size, output_dim)

    def backward(self, dout: float, optimizer: GradientDescent):

        x, z = self._cache # get the inputs of the previous layer

        da = self.activation_function.backward(z) # derivative of activation layer
        dw = numpy.transpose(x) * da * dout # chain rule (dout =: upstream gradient)

        self.weights = optimizer.optimize(weights=self.weights, gradient=dw)

        return dw











