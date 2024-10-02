import random
from math import pi

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import keras


class PINN_ho(keras.Model):
    """Physics Inspired Neural Network (PINN) for the harmonic oscillator. This model leverages
    knowledge of the governing equation:

        u_xx + u = 0

    to improve inferences outside of the training domain. The governing equation is evaluated using
    automatic differentiation via `tf.GradientTape`, and its residual is used as a regularization
    term in the loss.

    Parameters
    ----------
    layers : list[tuple[int, str]]
        Contains `(units, activation)` for each layer of the underlying sequential neural net.
    lb : float
        The lower bound of the domain to evaluate the governing equation residual.
    ub : float
        The uppder bound of the domain to evaluate the governing equation residual.
    alpha : float
        The weight for the governing equation regularization.
    """
    def __init__(self, layers: list[tuple[int, str]], ub: float, lb: float, alpha: float = 1.0):
        super().__init__()
        self._sequential = keras.Sequential([
            keras.layers.Dense(units, activation)
            for units, activation in layers
        ])
        self.lb = lb
        self.ub = ub
        self.x_res = tf.reshape(tf.linspace(lb, ub, 50), shape=(50, 1))
        self.alpha = alpha
        self._loss_tracker = keras.metrics.Mean(name='loss')
        self._mse_tracker = keras.metrics.Mean(name='mse')
        self._eqn_res_tracker = keras.metrics.Mean(name='eqn_res')

    def call(self, x, training: bool = False):  # type: ignore
        """Makes the prediction, with input normalized to [-1, 1]."""
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) + 1.0
        return self._sequential(x)

    def compute_derivatives(self, x):
        """Evaluate the 0th, 1st, and 2nd derivatives of the neural network."""
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                u = self(x)
            u_x = t2.gradient(u, x)
        u_xx = t1.gradient(u_x, x)
        return u, u_x, u_xx

    def compute_loss(self, x, y, y_pred, sample_weight=None):  # type: ignore
        """Computes the loss function, regularized by the governing equation's residual."""
        u, u_x, u_xx = self.compute_derivatives(self.x_res)
        eqn_residual = tf.reduce_mean(tf.square(
            u_xx + u
        ))

        mse = keras.losses.mean_squared_error(y, y_pred)
        loss = (
            mse
            + self.alpha * eqn_residual
        )

        self._loss_tracker.update_state(loss)
        self._mse_tracker.update_state(mse)
        self._eqn_res_tracker.update_state(eqn_residual)
        return loss

    @property
    def metrics(self):
        """Used for printing the epoch losses while training."""
        return [self._loss_tracker, self._mse_tracker, self._eqn_res_tracker]


if __name__ == '__main__':
    # Fabricate some data
    lb, ub = 0, 4 * pi
    x = np.linspace(lb, ub, 250)
    u = np.cos(x)  # exact solution to x'' + x = 0

    # Randomly sample the data
    n_samples = 20
    idx = random.sample(list(range(250)), n_samples)
    x_train = tf.constant(x[idx], dtype=tf.float32, shape=(n_samples, 1))
    u_train = tf.constant(u[idx], dtype=tf.float32, shape=(n_samples, 1))

    # Train model
    epochs = 5000
    layers = [
        (50, 'tanh'),
        (50, 'tanh'),
        (1, 'linear'),
    ]
    pinn = PINN_ho(layers, lb, ub, alpha=1.0)
    pinn.compile(optimizer='adam')
    pinn.fit(x_train, u_train, epochs=epochs)

    # Make prediction from trained model
    u_pred, u_x, u_xx = pinn.compute_derivatives(tf.constant(x, dtype=tf.float32, shape=(250, 1)))

    # Plot results
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=u, name='Truth', line_color='blue')
    )
    fig.add_trace(
        go.Scatter(
            x=x_train[:, 0], y=u_train[:, 0],
            mode='markers', marker=dict(symbol='x', color='black'),
            name='Train',
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=u_pred[:, 0], name='Fit')
    )
    fig.add_trace(
        go.Scatter(x=x, y=u_xx[:, 0] + u_pred[:, 0], name='Residual')
    )
    fig.update_layout(
        template='simple_white', width=1000, height=800,
    )
    fig.show()
