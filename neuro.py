import numpy as np
import scipy.special
import neuro_mnist as mnist


class NeuralNetwork:
    def __init__(self, inputNodes, outputNodes, hiddenNodes, learningRate):
        self.inodes = inputNodes
        self.onodes = outputNodes
        self.hnodes = hiddenNodes
        self.lr = learningRate
        self.activation_function = lambda x: scipy.special.expit(x)
        self.wih = np.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * \
            np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                   np.transpose(hidden_outputs))
        self.wih += self.lr * \
            np.dot((hidden_errors * hidden_outputs *
                    (1.0 - hidden_outputs)), np.transpose(inputs))
# --------------------------------------------------------------------------------------------


def trainNetwork(network):
    trainingData = mnist.trainingData()
    for epoch in range(3):
        for record in trainingData:
            inputs = mnist.rescaleInput(mnist.toFloat(record[1:]))
            targets = np.zeros(network.onodes) + 0.01
            targets[int(record[0])] = 0.99
            network.train(inputs, targets)


def testNetwork(network):
    scorecard = []
    testData = mnist.testData()
    for record in testData:
        inputs = mnist.rescaleInput(mnist.toFloat(record[1:]))
        target = int(record[0])
        outputs = network.query(inputs)
        label = np.argmax(outputs)
        # print(label)
        scorecard.append(label == target)
    return scorecard


def main():
    network = NeuralNetwork(inputNodes=784, outputNodes=10,
                            hiddenNodes=200, learningRate=0.1)
    trainNetwork(network)
    results = testNetwork(network)
    print('accuracy: ', (100 * len(list(filter(lambda x: x, results))) / 10000), '%')


if __name__ == "__main__":
    main()
