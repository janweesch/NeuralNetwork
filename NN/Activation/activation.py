import numpy


def select_activation(activation: str):

    match activation:

        case "sigmoid":
            return sigmoid

        case _:
            return "No activation function with this name"

def sigmoid(z_input: numpy.ndarray):

    a = 1 / (1 + numpy.exp(-z_input))

    return a
