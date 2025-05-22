from make_dataset import generate_dataset
import matplotlib.pyplot as plt
import numpy as np

def forward_pass():
    W1 = np.random.randn(100, 3) * 0.01
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 3) * 0.01
    b2 = np.zeros((1, 3))

    Z1 = X.dot(W1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    output = Z2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    X, y = generate_dataset(n_per_class=10, noise=0)
    for i in range(len(X)):
        print(f"Label: {y[i]}")
        print(f"Value: {X[i]}")
        plt.imshow(X[i].reshape(10, 10), cmap='gray')
        plt.show()