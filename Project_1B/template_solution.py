# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import os

def transform_features(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component in a given row of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: matrix of floats: dim = (700,21), transformed input with 21 features
    """
    n_samples = X.shape[0]
    X_transformed = np.zeros((n_samples, 21))

    # Fill columns 0-4 with the original linear features x1, ..., x5
    X_transformed[:, 0:5] = X

    # Fill columns 5-9 with the quadratic terms x1^2, ..., x5^2
    X_transformed[:, 5:10] = X ** 2

    # Fill columns 10-14 with the exponential terms exp(x1), ..., exp(x5)
    X_transformed[:, 10:15] = np.exp(X)

    # Fill columns 15-19 with the cosine terms cos(x1), ..., cos(x5)
    X_transformed[:, 15:20] = np.cos(X)

    # Last column is the constant feature, which acts as the bias term after training
    X_transformed[:, 20] = 1.0

    assert X_transformed.shape == (n_samples, 21)
    return X_transformed


def fit_logistic_regression(X, y):
    """
    This function receives training data points, transforms them, and then fits the logistic regression on this 
    transformed data. Finally, it outputs the weights of the fitted logistic regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of integers \in {0,1}, dim = (700,), input labels

    Returns
    ----------
    weights: array of floats: dim = (21,), optimal parameters of logistic regression
    """
    # Initialize the 21 regression weights to zero before gradient descent
    weights = np.zeros((21,))
    X_transformed = transform_features(X)

    # Sigmoid maps logits to probabilities in (0, 1)
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # Standardize only the first 20 features to improve optimization stability;
    # the constant feature must remain equal to 1 so it can represent the bias
    X_mean = X_transformed[:, :-1].mean(axis=0)
    X_std = X_transformed[:, :-1].std(axis=0)
    X_std[X_std == 0] = 1.0

    X_scaled = X_transformed.copy()
    X_scaled[:, :-1] = (X_scaled[:, :-1] - X_mean) / X_std

    # These values control the size of each update, the number of updates,
    # and the strength of L2 regularization
    learning_rate = 0.1
    n_iterations = 20000
    l2_lambda = 1e-3
    n_samples = X_scaled.shape[0]

    # Perform full-batch gradient descent on the regularized logistic loss
    for _ in range(n_iterations):
        logits = X_scaled @ weights
        probs = sigmoid(logits)

        # For logistic regression with cross-entropy loss, probs - y is the core error term
        error = probs - y
        gradient = (X_scaled.T @ error) / n_samples

        # Add the gradient of the L2 penalty to discourage overly large weights;
        # the bias is excluded from regularization
        reg_gradient = (l2_lambda / n_samples) * weights
        reg_gradient[-1] = 0.0
        gradient += reg_gradient

        weights -= learning_rate * gradient

    # Training was done on standardized features, but submission requires weights
    # for the original feature space, so we undo the scaling here
    weights_original = np.zeros_like(weights)
    weights_original[:-1] = weights[:-1] / X_std
    weights_original[-1] = weights[-1] - np.sum((weights[:-1] * X_mean) / X_std)

    weights = weights_original
    assert weights.shape == (21,)
    return weights


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(script_dir, "train.csv"))
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # Fit the logistic regression model and obtain the final 21 coefficients
    w = fit_logistic_regression(X, y)
    # Save one weight per line, exactly as required for submission
    np.savetxt(os.path.join(script_dir, "results.csv"), w, fmt="%.12f")
