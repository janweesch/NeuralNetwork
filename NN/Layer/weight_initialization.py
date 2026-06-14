import numpy as np
from NN.Activation.activation import ActivationFunction, ReLU, Sigmoid, Tanh, Affine

def weight_initialization(input_dim: int, output_dim: int, activation_function: ActivationFunction) -> np.ndarray:
    """
    Selects the appropriate weight initialization depending on the activation function.

    Args:
        input_dim: Number of neurons of the previous layer
        output_dim: Number of neurons of the current layer
        activation_function: Activation function of the current layer

    Returns:
        np.ndarray: Initialized weights
    """

    if activation_function == ReLU:
        # He initialization
        weights = kaiming(input_dim=input_dim, output_dim=output_dim)

    elif activation_function in (Sigmoid, Tanh):
        # Xavier initialization
        weights = xavier(input_dim=input_dim, output_dim=output_dim)

    else:
        # Simple random initialization
        weights = random_weight_initialization(input_dim, output_dim)

    return weights


def random_weight_initialization(input_dim: int, output_dim: int) -> np.ndarray:
    """
    Initialise weights of layer from a random uniform distribution.

    Args:
        input_dim: Number of neurons of the previous layer
        output_dim: Number of neurons of the current layer

    Returns:
        np.ndarray: Initialized weights
    """

    init_weights = np.random.rand(input_dim, output_dim)

    return init_weights


def xavier(input_dim: int, output_dim: int) -> np.ndarray:
    """
    Initializes the weights of a layer with the xavier initialization. The xavier initialization is used for the tanh activation function.

    Args:
        input_dim: Number of neurons within the previous layer
        output_dim: Number of neurons within the current layer

    Returns:
        np.ndarray: Initialized weights
    """

    init_weights = np.random.normal(loc=0.0, scale=np.sqrt(1/input_dim), size=(input_dim, output_dim))

    return init_weights

def kaiming(input_dim: int, output_dim: int) -> np.ndarray:
    """
    Initializes the weights of a layer with the kaiming initialization. The kaiming initialization is preferably used for the ReLU activation function.

    Args:
    prev_layer_neurons: Number of neurons within the previous layer
    curr_layer_neurons: Number of neurons within the current layer

    Returns:
        np.ndarray: Initialized weights
    """

    init_weights = np.random.normal(loc=0.0, scale=np.sqrt(2/input_dim), size=(input_dim, output_dim))

    return init_weights