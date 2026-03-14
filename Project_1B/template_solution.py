# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)

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

    # Linear features
    X_transformed[:, 0:5] = X

    # Quadratic features
    X_transformed[:, 5:10] = X ** 2

    # Exponential features
    X_transformed[:, 10:15] = np.exp(X)

    # Cosine features
    X_transformed[:, 15:20] = np.cos(X)

    # Constant feature
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
    weights = np.zeros((21,))
    X_transformed = transform_features(X)

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # Standardize all features except the constant one
    X_mean = X_transformed[:, :-1].mean(axis=0)
    X_std = X_transformed[:, :-1].std(axis=0)
    X_std[X_std == 0] = 1.0

    X_scaled = X_transformed.copy()
    X_scaled[:, :-1] = (X_scaled[:, :-1] - X_mean) / X_std

    # Hyperparameters
    learning_rate = 0.1
    n_iterations = 20000
    l2_lambda = 1e-3
    n_samples = X_scaled.shape[0]

    # Gradient descent
    for _ in range(n_iterations):
        logits = X_scaled @ weights
        probs = sigmoid(logits)

        error = probs - y
        gradient = (X_scaled.T @ error) / n_samples

        # L2 regularization on all weights except the bias term
        reg_gradient = (l2_lambda / n_samples) * weights
        reg_gradient[-1] = 0.0
        gradient += reg_gradient

        weights -= learning_rate * gradient

    # Convert weights back to the original feature space
    weights_original = np.zeros_like(weights)
    weights_original[:-1] = weights[:-1] / X_std
    weights_original[-1] = weights[-1] - np.sum((weights[:-1] * X_mean) / X_std)

    weights = weights_original
    assert weights.shape == (21,)
    return weights


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit_logistic_regression(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
