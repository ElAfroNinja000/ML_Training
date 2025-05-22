import numpy as np
import random


def add_noise(image, noise_level=0.1):
    noisy = image.copy()
    n_pixels = int(noise_level * image.size)
    indices = random.sample(range(image.size), n_pixels)
    for idx in indices:
        i, j = divmod(idx, image.shape[1])
        noisy[i, j] = 1 - noisy[i, j]  # flip pixel
    return noisy


def generate_square():
    img = np.zeros((10, 10))
    img[2:8, 2:8] = 1
    return img


def generate_circle():
    img = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if ((i - 5) ** 2 + (j - 5) ** 2) < 16:
                img[i, j] = 1
    return img


def generate_triangle():
    img = np.zeros((10, 10))
    for i in range(5, 10):
        start = 5 - (i - 5)
        end = 5 + (i - 5)
        img[i, start:end + 1] = 1
    return img


# Génération du dataset
def generate_dataset(n_per_class=10, noise=0.1):
    X = []
    y = []

    for _ in range(n_per_class):
        X.append(add_noise(generate_square(), noise).flatten())
        y.append([1, 0, 0])  # square

        X.append(add_noise(generate_circle(), noise).flatten())
        y.append([0, 1, 0])  # circle

        X.append(add_noise(generate_triangle(), noise).flatten())
        y.append([0, 0, 1])  # triangle

    X = np.array(X)
    y = np.array(y)
    return X, y


# Exemple d'utilisation
X, y = generate_dataset(n_per_class=10, noise=0.1)
print("X shape:", X.shape)
print("y shape:", y.shape)
