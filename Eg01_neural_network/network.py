"""
The network in bool `Neural Network and Deep Learning` Chapter01
"""

import numpy as np
import random


def sigmoid(z):
    """The sigmoid function
    """
    return 1. / (1. + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if a in input"""
        for b, w in zip(self.bias, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent

        Args:
            training_data ([type]): a list of tuples (x, y) representing the training inputs and the desired outputs
            epochs ([type]): [description]
            mini_batch_size ([type]): [description]
            eta ([type]): [description]
            test_data ([type], optional): If test_data is provides then the network will be evaluated against the test 
            data after each epoch. This is useful for tracking progress, but slows things down substantially
        """
        test_data = list(test_data)
        training_data = list(training_data)
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete')

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the networks's weights and biases by applying gradient descent using backpropagation to single mini batch

        Args:
            mini_batch: A list of tuples (x, y)
            eta (float): learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b - eta / len(mini_batch) * nb for b, nb in zip(self.bias, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C(x)

        Args:
            x ([type]): [description]
            y ([type]): [description]
        """
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outpus the correct result

        Args:
            test_data ([type]): [description]
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        [summary]

        Args:
            output_activations ([type]): [description]
            y ([type]): [description]
        """
        return output_activations - y
