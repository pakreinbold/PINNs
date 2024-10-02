# PINNs
Physics-Informed Neural Networks

This repository contains recreations of the work performed by Raissi et al. (2019). Certain code snippets from said work are imported directly, or used as starting points for the code here.

## Harmonic Oscillator
The first file is a short proof-of-principle on the simplest physical example: the harmonic oscillator. The governing equation for this system is 

![u'' + u = 0,](https://latex.codecogs.com/svg.image?\bg{black}u''&plus;u=0)

The exact solution used for training is

![u(x) = cos(x).](https://latex.codecogs.com/gif.latex?u%28x%29%20%3D%20%5Ccos%28x%29%2C%20%5C%3A%5C%3A%20x%5Cin%5B0%2C%204%5Cpi%29)

Which is sampled at a number (N ~ 20) of random locations. The loss function penalizes the deviation of the neural network output form these "observations". Additionally, the loss function contains a regularization term that seeks to enforce the governing equations at a denser grid (N ~ 50). Crucially, the governing equations can be evaluated by using automatic differentiation via Tensorflow's GradientTape() to obtain the required derivative(s) from the neural network. The weights & biases of the neural network are trained using Keras's Adam optimizer.

<img src="harmonic_oscillator.png" alt="drawing" width="750"/>

## Korteweg-de Vries Equation
The governing equation is 

![](https://latex.codecogs.com/gif.latex?%5Cpartial_t%20u%20&plus;%20%5Cpartial_%7Bxxx%7Du%20&plus;%206%20u%20%5Cpartial_xu%20%3D%200)

The exact solution used for training is 

![u(x,t) = 1/2 sech^2(1/2(x - t))](https://latex.codecogs.com/gif.latex?u%28x%2Ct%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Ctext%7Bsech%7D%5E2%28%5Cfrac%7B1%7D%7B2%7D%28x%20-%20t%29%29%2C%20%5C%3A%5C%3A%5C%3A%20x%5Cin%5B0%2C10%5D%2C%20t%5Cin%5Ctimes%5B0%2C10%5D)

Again, the loss function fits the neural network to N ~ 150 observations (split between an initial condition and randomly sampled throughout the bulk), and satisfying the governing equations is used as a constraint.


References:

Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.

Equations generated using https://www.codecogs.com/latex/eqneditor.php
