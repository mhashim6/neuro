import numpy as np
import matplotlib.pyplot as plt


def trainingData():
    return fileLines("mnist_train.csv")


def testData():
    return fileLines("mnist_test.csv")


def fileLines(fileName):
    file = open(fileName, 'r')
    lines = file.readlines()
    file.close()
    return np.array(lines)


def toFloat(string_input):
    # [1:len(input)-1] => omit both first and last elements, as they are non-numerical strings.
    return np.asfarray(string_input.split(',')[1:len(string_input)-1], dtype=str)


def rescaleInput(input):
    """
    scales input's elements down to range 0.1 to 1.0.
    """
    return (0.99 * input / 250.0) + .1
