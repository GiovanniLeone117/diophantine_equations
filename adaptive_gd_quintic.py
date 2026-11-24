"""
This file implements different variants adaptive gradient descent algorithms for solving continuous 5th
degree polynomials in the form of p(x) = a_1x_1^5 + | - a_2x_2^4.... + | - ... a_nx_n^1 + | - c = 0
where a_n, x_n and c are all real numbers. The goal is to find the roots of the polynomial
where p(x) = 0. Continuous is defined as the polynomial being differentiable at all points in its domain.
"""

import numpy as np
import math

# Coefficients for the quintic equation
COEFFICIENTS = [1, 0, 0, 0, -1, 0.6] # The real valued root solutions is x = -1.1
X = 1.4 # Initial guess for x

# solution = 1

# Converge if f(x) is this close to 0
tolerant_residual = 1e-10

# HYPERPARAMETERS
LEARNING_RATE = 0.005
NUM_ITERATIONS = 5000
BETA1 = 0.90 # This hyperparameter is used to calculate the weighted moving average of previous gradients
BETA2 = 0.90

OPTIMIZER = "GD_MOMENTUM"


def loss_calculation(x: float):
    """
    The loss function is defined as p(x) squared. Where p(x) is the quintic equation.
    :return: loss: the loss value for the current x
    """
    p_x = np.polyval(COEFFICIENTS, x)
    loss = p_x ** 2
    return loss, p_x


def gradient_calculation(x: float):
    """
    Using the chain rule, the derivative of the loss function with respect to x is defined as
    2 * p(x) * p'(x) where p'(x) is the derivative of the quintic equation. This function
    will calculate the gradient of the loss function with respect to x. Since this is a
    single dimensional optimization problem, the gradient is just defined as the derivative
    of the loss function with respect to x.
    """
    p_x = np.polyval(COEFFICIENTS, x)
    p_prime_x = np.polyval(np.polyder(COEFFICIENTS), x)

    # Bound the gradient to avoid exploding slopes
    gradient = 2 * p_x * p_prime_x
    # gradient = np.clip(gradient, -200, 200)
    return gradient


def first_moment_calculation(gradient, mt, iteration):
    """
    Calculate the weighted moving average of the previous gradients
    with bias correction (first moment)
    """
    new_mt = (BETA1 * mt) + (1 - BETA1) * gradient
    bias_corrected_mt = new_mt / (1 - BETA1**iteration)
    return bias_corrected_mt, new_mt


def second_moment_calculation(gradient, vt, iteration):
    """
    Calculate the weighted moving average of the previous squared gradients
    with bias correction (second moment)
    """
    new_vt = (BETA2 * vt) + (1 - BETA2) * (gradient ** gradient)
    bias_corrected_vt = new_vt / (1 - BETA2**iteration)
    return bias_corrected_vt, new_vt


def update_x(current_x, gradient, first_moment, second_moment, gradient_sum, epsilon = 1e-8):
    """
    Update the value of x by subtracting the product of the learning rate and momentum
    from the current x
    """
    # Use Adam parameter update rule
    if OPTIMIZER == "ADAM":
        return current_x - LEARNING_RATE * (first_moment / (math.sqrt(second_moment) + epsilon))

    # Use RMSProp parameter update rule
    elif OPTIMIZER == "RMSProp":
        return current_x - LEARNING_RATE * (gradient / (math.sqrt(second_moment) + epsilon))

    # Use AdaGrad parameter update rule
    elif OPTIMIZER == "ADAGRAD":
        return current_x - LEARNING_RATE # TODO finish this update rule, have to keep track of the cumulative sum of square gradients

    elif OPTIMIZER == "GD_MOMENTUM":
        # Use momentum only to calculate the parameter update
        return current_x - LEARNING_RATE * first_moment


def gradient_descent(x: float, uncorrected_mt, uncorrected_vt):
    """
    Run the gradient descent algorithm for a specified number of iterations.
    Each iteration will adjust the value of x to achieve optimization
    of the loss function.
    """
    for i in range(1, NUM_ITERATIONS):
        loss, p_x = loss_calculation(x)
        gradient = gradient_calculation(x)
        corrected_mt, uncorrected_mt = first_moment_calculation(gradient, uncorrected_mt, i)
        #TODO call vt_calculation

        print(f"Iteration {i + 1}: Loss = {loss}, Gradient = {gradient}, x = {x}")
        x = update_x(corrected_mt, x)
        if abs(p_x) <= tolerant_residual:
            print("Converged")
            break

def main():

    """
    # Test one iteration
    loss = loss_calculation(X)
    gradient = gradient_calculation()
    print(loss, gradient)
    new_x = update_x(gradient, X)
    print(new_x)
    """
    gradient_descent(X, 0, 0)


if __name__ == "__main__":
    main()
