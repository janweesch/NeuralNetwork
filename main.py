from NN.Network.network import Network
import numpy


myNetwork = Network(2, 1, 0, [], [], "sigmoid")



print(myNetwork.forward(x_input=numpy.array([-2, -1])))