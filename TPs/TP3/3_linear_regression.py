import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def format_data():
    with open('artworks.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array(list(reader))[:, [1, 2]]
    labels = data[0]
    data = np.delete(data, 0, 0)
    data = np.array(data, dtype=int)
    return labels, data

def split_data(data):
    X, y = data[:, 0], data[:, 1]
    X = X.astype(float)
    y = y.astype(float)
    data = np.column_stack([X, y])
    training_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return training_data, test_data

def linear_regression(training_data):
    X, y = training_data[:, 0], training_data[:, 1]
    X = np.column_stack([X, np.ones(len(X))])
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def test_linear_regression(theta, training_data, test_data, data):
    X, y = data[:, 0], data[:, 1]
    X_train, y_train = training_data[:, 0], training_data[:, 1]
    X_test, y_test = test_data[:, 0], test_data[:, 1]

    y_pred = X_test.dot(theta[1]) + theta[0]
    mse = mean_squared_error(y_test, y_pred)

    print(f"Test MSE : {mse}")
    plt.scatter(X, y)
    X_line = np.linspace(X.min(), X.max(), 100)
    X_line_with_bias = np.column_stack([X_line, np.ones(len(X_line))])
    y_line = X_line_with_bias.dot(theta)
    plt.plot(X_line, y_line, color='red', label='Droite de régression')
    plt.xlabel('Année')
    plt.ylabel('Moyenne du rouge')
    plt.title('Régression Linéaire : Moyenne du Rouge en Fonction de l\'Année')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    labels, data = format_data()
    training_data, test_data = split_data(data)
    theta = linear_regression(training_data)
    test_linear_regression(theta, training_data, test_data, data)
