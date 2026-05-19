

from NN.Activation.activation import Sigmoid
from NN.Loss.loss import BCE
from NN.Network.network import Network
import numpy
from NN.Optimization.optimizer import GradientDescent
from NN.Training.train import Train


myNetwork = Network(2, 1, 1, [2], [Sigmoid], output_layer=1, output_activation=Sigmoid)

#x_input = numpy.array([-2, -1]).reshape((1, 2))
#print(x_input.shape)
#print(myNetwork.forward(x_input=x_input))


training_data = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_truth = numpy.array([[0], [1], [1], [0]])

trainer = Train(myNetwork, loss=BCE, optimizer=GradientDescent)

trained_network = trainer.train(training_data=training_data, y_truth=y_truth, epochs=2, batch_size=None)

