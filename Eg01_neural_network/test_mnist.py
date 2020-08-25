"""
Use network to training MNIST
"""

import mnist_loader
import network


def test01():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 100, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)


if __name__ == "__main__":
    test01()
