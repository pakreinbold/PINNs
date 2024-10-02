from typing import Callable

import numpy as np
import tensorflow as tf


def initialize_NN(layers: list[int]) -> tuple[list[tf.Variable], list[tf.Variable]]:
    """Use Xavier initialization to create the weights and biases for a neural network.

    Parameters
    ----------
    layers : list[int]
        Contains the number of neurons in each layer.

    Returns
    -------
    weights : list[tf.Variable]
        Contains the matrix weights for each layer.
    biases : list[tf.Variable]
        Contains the bias vectors for each layer.
    """
    weights = []
    biases = []
    num_layers = len(layers)
    for i in range(0, num_layers - 1):
        W = xavier_init((layers[i], layers[i + 1]))
        b = tf.Variable(tf.zeros([1, layers[i + 1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases


def xavier_init(size: tuple[int, int]) -> tf.Variable:
    """Initializes the weights of a neural network according to the scheme discussed in [1].

    Parameters
    ----------
    size : tuple[int, int]
        The input and output dimensions of the layer.

    Returns
    -------
    tf.Variable
        The weight matrix.

    References
    ----------
    .. [1] Glorot, Xavier, and Yoshua Bengio. Understanding the difficulty of training deep
        feedforward neural networks. Proceedings of the thirteenth international conference on
        artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.
    """
    in_dim, out_dim = size
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(
        tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
        dtype=tf.float32
    )


def get_gradient(x: tf.Variable, fun: Callable):
    """Take the derivative of function fun(x) wrt its only argument x"""
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fun(x)

    return tape.gradient(y, x)


def take_derivative(fun: Callable, order: int = 1) -> Callable:
    if order < 1:
        return fun
    else:
        order -= 1
    return take_derivative(lambda x: get_gradient(x, fun), order=order)
