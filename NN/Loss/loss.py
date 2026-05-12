import numpy


class Loss:

    def forward(self, y_out, y_truth):
        return NotImplementedError

    def backward(self, y_out, y_truth):
        return NotImplementedError

class L1(Loss):

    def forward(self, y_out, y_truth):

        loss = - numpy.abs(y_out - y_truth)
        average_loss = numpy.mean(loss)

        return average_loss

    def backward(self, y_out, y_truth):

        dL = None

        dL_1 = numpy.where(y_out > y_truth, y_out,  1)

        dL_2 = numpy.where(y_out > y_truth, )

        
        assert dL is not None, "L1 Loss Gradient can not be None!"

        return dL

class BCE(Loss):

    def forward(self, y_out, y_truth):

        loss = - (y_truth * numpy.log(y_out) + (1-y_truth) * (1-numpy.log(y_out))) # binary cross entropy
        average_loss = numpy.mean(loss) # accumulated loss

        return average_loss

    def backward(self, y_out, y_truth):

        # Don`t sum up the gradient, one gradient for each sample.
        # 1/N in front of the loss is necessary to average the loss without changing the learning Rate for every Batch

        dL = -1/len(y_out) * (y_truth/y_out + (1-y_truth)/(1-y_out)) # derivative of dL

        return dL


