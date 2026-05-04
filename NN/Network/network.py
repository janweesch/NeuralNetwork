import numpy
from typing import List
from NN.Layer.layer import Layer
from NN.Activation.activation import select_activation


class Network:

    def __init__(self, x_input: int, classes: int,  hidden_layers: int, hidden_sizes: List[int], activations: List[str], output_activation: str ):

        assert hidden_layers == len(hidden_sizes) and hidden_layers == len(activations), "Length of hidden_layers list must be equal to length of hidden_size list!"

        prev_size = x_input
        current_layer = None

        """Initialize the Neural Network, with all hidden layers and corresponding activation functions."""
        for index, hidden_size, activation in enumerate(zip(hidden_sizes, activations)):
            activation = select_activation(activation) # select the activation function

            previous_layer = current_layer
            current_layer = Layer(input_dim=prev_size, output_dim=hidden_size, activation_function=activation, previous_layer=previous_layer) #initialize layer
            previous_layer.next = current_layer
            if index == 0:
                self.network = current_layer

            prev_size = hidden_size # overwrite previous size

        output_activation = select_activation(output_activation)

        if not current_layer:
            self.network = Layer(input_dim=prev_size, output_dim=classes, activation_function=output_activation, previous_layer=current_layer) # output Layer
        else:
            Layer(input_dim=prev_size, output_dim=classes, activation_function=output_activation, previous_layer=current_layer)  # output Layer



    def forward(self, x_input: numpy.ndarray):

        layer = self.network
        output = x_input
        while layer:
            output = layer.forward(x_input=output)
            layer = layer.next_layer

        return output











