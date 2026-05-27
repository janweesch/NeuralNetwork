from __future__ import annotations
import numpy as np
from NN.Activation.activation import Sigmoid, Affine
from NN.Optimization.optimizer import Optimizer, GradientDescent
from NN.Data.visualizer import Visualizer


class Layer:
    """
    Layer with neurons and activation functions.

    Attributes:
         input_dim: Number of neurons of the previous layer
         output_dim: Number of neurons of the current layer
         weights: Weight matrix of shape (input_dim+1, output_dim), containing bias
         activation_function: Activation function of the layer.
         _cache: Stores the input and output of activation_function for backpropagation
    
    Args:
        input_dim: Number of neurons of the previous layer
        output_dim: Number of neurons of the current layer
        activation_function: Activation function of the layer, default is Affine layer
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1, activation_function = Affine):

        self.input_dim = input_dim # dimension (number of Neurons) of previous Layer
        self.output_dim = output_dim # dimension (number of Neurons) of current Layer

        self.weights = None
        self.random_weight_initialization(input_dim=input_dim, output_dim=output_dim)  # include bias

        self.activation_function = activation_function # activation function of layer

        self._cache = None

    def random_weight_initialization(self, input_dim: int, output_dim: int):
        """
        Initialise weights of layer from a random uniform distribution.
        
        Args:
            input_dim: Number of neurons of the previous layer
            output_dim: Number of neurons of the current layer
        """
        
        self.weights = np.random.rand(input_dim+1, output_dim)
        #self.weights = np.random.uniform(-1, 1, size=(input_dim+1, output_dim))

    def forward(self, x_input : np.ndarray) -> np.ndarray:
        """
        Pass the input from the previous layer through the current layer.
        
        Args:
            x_input: Input, coming from the previous layer

        Returns:
            a: Result after activation_function
        """
        x = np.concatenate((np.ones((x_input.shape[0], 1)), x_input), axis=1)

        assert self.weights.shape[0] ==  x.shape[1], "Weights and Feature input does not have the same dimension!"
        z = x @ self.weights # preactivation
        a = self.activation_function.forward(z) # call the forward pass in the activation
        #if a.shape[1] == 2:
         #   Visualizer.visualize_2d_points(a, title= "X-OR Data Transformed")
        self._cache = (x, z) # store the output
        return a # shape (Batch_size, output_dim)

    def backward(self, dout: float, optimizer: GradientDescent):
        """
        Passes the Gradient through the layer, updating the weights.

        Args:
            dout: Gradient, received from the previous layer
            optimizer: Updates the weights

        Returns:
            dx: Gradient for the next layer
        """

        x, z = self._cache # get the inputs of the previous layer

        dz = self.activation_function.backward(z)  # derivative of activation layer
        dw = x.T @ (dz * dout)# chain rule (dout =: upstream gradient)

        # remove bias
        weights_o_bias = np.delete(self.weights, 0, axis=0)
        dx = (dz * dout) @ weights_o_bias.T

        self.weights = optimizer.optimize(weights=self.weights, gradient=dw)

        return dx











