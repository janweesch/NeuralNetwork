import numpy as np
from typing import List, Optional
from NN.Layer.layer import Layer
from NN.Activation.activation import ActivationFunction, Affine
from NN.Optimization.optimizer import Optimizer


class Network:
    """
    Fully connected feedforward neural network.

    Supports multiple hidden layers with configurable
    activation functions.

    Attributes:
        layers: List of network layers

    Args:
        input_dim: Number of input neurons
        classes: Number of different classes in the dataset
        hidden_layers: Number of hidden layers within the Neural Network
        hidden_sizes: Number of neurons in each hidden layer
        activations: Activation function for each hidden layer
        output_activation_function: Activation function of the last layer (output layer)
    """
    def __init__(self, input_dim: int, classes: int,  hidden_layers: int, hidden_sizes: List[int], activations: List[ActivationFunction], output_activation_function=Affine):

        assert hidden_layers == len(hidden_sizes) and hidden_layers == len(activations), "Length of hidden_layers list must be equal to length of hidden_size list and activations list!"

        prev_size = input_dim
        self.layers = [] # store all layers

        """Initialize the Neural Network, with all hidden layers and corresponding activation functions."""
        for hidden_size, activation in zip(hidden_sizes, activations):

            current_layer = Layer(input_dim=prev_size, output_dim=hidden_size, activation_function=activation) # initialize layer
            self.layers.append(current_layer)

            prev_size = hidden_size # current_layer is previous_layer for the next layer

        output_layer_units = classes if classes > 2 else 1 # use one output neuron, reduce computational cost
        self.layers.append(Layer(input_dim=prev_size, output_dim=output_layer_units, activation_function=output_activation_function))

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """
        Passes the input through each layer.

        Args:
            x_input: Input to be passed through the layer

        Returns:
            np.ndarray: Predicted values for each x_input sample. Logits are already scaled with the activation_function of the output_layer
        """
        y_out = x_input
        for layer in self.layers:
            y_out = layer.forward(x_input=y_out)


        return y_out

    def backward(self, gradient: float, optimizer: Optimizer):
        """
        Passes the Gradient through the complete network updating the weights

        Args:
            gradient: Gradient of the Loss w.r.t y_out (output of forward_pass)
            optimizer: Responsible for updating the weights
        """
        dout = gradient
        for layer in self.layers[::-1]:
            dout = layer.backward(dout, optimizer)























