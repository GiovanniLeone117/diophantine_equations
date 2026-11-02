"""
This file implements a gradient descent algorithm for solving continuous 5th
degree polynomials in the form of p(x) = a_1x_1^5 + | - a_2x_2^4.... + | - ... a_nx_n^1 + | - c = 0
where a_n, x_n and c are all read numbers. The goal is to find the roots of the polynomial
where p(x) = 0. Continuous is defined as the polynomial being differentiable at all points in its domain.
"""
import numpy as np

# Coefficients for the quintic equation
COEFFICIENTS = [1, 0, 0, 0, -1, 0] # The real valued root solutions is x = 1, x = -1, x = 0
X = 0.5 # Random initial guess for x

# HYPERPARAMETERS
#TODO this is a fixed learning rate, adjust it to be adaptive to the gradient to prevent overshooting the minimum of the loss function.
LEARNING_RATE = 0.001
NUM_ITERATIONS = 5000


def loss_calculation(x: float):
    """
    The loss function is defined as p(x) squared. Where p(x) is the quintic equation.
    :return: loss: the loss value for the current x
    """
    p_x = np.polyval(COEFFICIENTS, x)
    loss = p_x ** 2
    return loss


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
    gradient = np.clip(gradient, -200, 200)
    return gradient


def update_x(gradient, current_x):
    """
    Update the value of x by subtracting the product of the learning rate and gradient
    from the current x
    """
    return current_x - LEARNING_RATE * gradient


def gradient_descent(x: float):
    """
    Run the gradient descent algorithm for a specified number of iterations.
    Each iteration will adjust the value of x to achieve optimization
    of the loss function.
    """
    for i in range(NUM_ITERATIONS):
        loss = loss_calculation(x)
        gradient = gradient_calculation(x)
        print(f"Iteration {i + 1}: Loss = {loss}, Gradient = {gradient}, x = {x}")
        x = update_x(gradient, x)


def main():

    """
    # Test one iteration
    loss = loss_calculation(X)
    gradient = gradient_calculation()
    print(loss, gradient)
    new_x = update_x(gradient, X)
    print(new_x)
    """
    gradient_descent(X)


if __name__ == "__main__":
    main()


