import numpy as np

def xavier(prev_layer_neurons: int, curr_layer_neurons: int) -> np.ndarray:
    """
    Initializes the weights of a layer with the xavier initialization. The xavier initialization is used for the tanh activation function.

    Args:
        prev_layer_neurons: Number of neurons within the previous layer
        curr_layer_neurons: Number of neurons within the current layer

    Returns:
        np.ndarray: Initialized weights
    """

    init_weights = np.random.normal(loc=0.0, scale=1/prev_layer_neurons, size=(prev_layer_neurons, curr_layer_neurons))

    return init_weights

def kaiming(prev_layer_neurons: int, curr_layer_neurons: int) -> np.ndarray:
    """
    Initializes the weights of a layer with the kaiming initialization. The kaiming initialization is preferably used for the ReLU activation function.

    Args:
    prev_layer_neurons: Number of neurons within the previous layer
    curr_layer_neurons: Number of neurons within the current layer

    Returns:
        np.ndarray: Initialized weights
    """

    init_weights = np.random.normal(loc=0.0, scale=2/prev_layer_neurons, size=(prev_layer_neurons, curr_layer_neurons))

    return init_weights