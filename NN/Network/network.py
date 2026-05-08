import numpy
from typing import List
from NN.Layer.layer import Layer
from NN.Activation.activation import ActivationFunction


class Network:

    def __init__(self, x_input: int, classes: int,  hidden_layers: int, hidden_sizes: List[int], activations: List[ActivationFunction], output_activation: ActivationFunction ):

        assert hidden_layers == len(hidden_sizes) and hidden_layers == len(activations), "Length of hidden_layers list must be equal to length of hidden_size list!"

        prev_size = x_input
        self.layers = [] # store all layers

        """Initialize the Neural Network, with all hidden layers and corresponding activation functions."""
        for index, hidden_size, activation in enumerate(zip(hidden_sizes, activations)):

            current_layer = Layer(input_dim=prev_size, output_dim=hidden_size, activation_function=activation) # initialize layer
            self.layers.append(current_layer)

            prev_size = hidden_size # overwrite previous size

        self.layers.append(Layer(input_dim=prev_size, output_dim=classes, activation_function=output_activation))  # output Layer

    def forward(self, x_input: numpy.ndarray):
        """Forward pass
        x_input: Input into array (N, D) (samples, dimension)
        """
        output = x_input
        for layer in self.layers:
            output = layer.forward(x_input=output)

        return output

    def backward(self, loss: float):
        """
        Backward pass
        """
        dout = loss
        for layer in self.layers[::-1]:
            dout = layer.backward(dout=dout)

        return dout





















