from NN.Data.visualizer import Visualizer
from NN.Activation.activation import Sigmoid, ReLU
from NN.Loss.loss import BCE, L1
from NN.Network.network import Network
import numpy
from NN.Optimization.optimizer import GradientDescent
from NN.Training.train import Train

myNetwork = Network(2, 2, 1, [2], [Sigmoid], output_activation_function=Sigmoid)

training_data = numpy.array([[0,0], [1, 1], [0, 1], [1, 0]])
y_truth = numpy.array([[0], [0], [1], [1]])

#print([layers.weights for layers in myNetwork.layers])

# overfitting
#training_data = numpy.array([[0, 1],[0, 0], [1, 0]])
#y_truth = numpy.array([[1], [0], [1]])

"""Visualization"""
Visualizer.visualize_2d_points(training_data, title= "X-OR Data")
print(myNetwork.forward(training_data))

trainer = Train(myNetwork, loss=BCE(), optimizer=GradientDescent())

trained_network = trainer.train(training_data=training_data, y_truth=y_truth, epochs= 10000)

#print([layers.weights for layers in trained_network.layers])

print(trained_network.forward(training_data))
print([0 if o < 0.5 else 1 for o in trained_network.forward(training_data)])


