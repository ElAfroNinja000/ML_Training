from generate_training_data import STYLES, OUTPUT_DIR

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(image_path):
    img = Image.open(image_path).resize((256, 256))
    img_np = np.array(img)
    hist_r = np.histogram(img_np[:, :, 0], bins=8, range=(0, 255))[0]
    hist_g = np.histogram(img_np[:, :, 1], bins=8, range=(0, 255))[0]
    hist_b = np.histogram(img_np[:, :, 2], bins=8, range=(0, 255))[0]
    features = np.concatenate([hist_r, hist_g, hist_b])
    return normalize([features])[0]

def extract_all_features():
    X, y, image_paths = [], [], []
    for style in STYLES:
        style_dir = os.path.join(OUTPUT_DIR, style)
        for filename in os.listdir(style_dir):
            path = os.path.join(style_dir, filename)
            X.append(extract_features(path))
            y.append(style)
            image_paths.append(path)
    return X, y, image_paths

def train(features, labels, image_paths):
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        features, labels, image_paths, test_size=0.4, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=1) 
    knn.fit(X_train, y_train)
    return knn, X_test, y_test, paths_test




if __name__ == "__main__":
    features, labels, image_paths = extract_all_features()
    knn, X_test, y_test, paths_test = train(features, labels, image_paths)

    y_pred = knn.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    for i in range(len(y_test)):
        if y_test[i] == "dots" and y_pred[i] != "dots":
            img = Image.open(paths_test[i])
            plt.imshow(img)
            plt.title(f"Vrai: {y_test[i]} — Prédit: {y_pred[i]}")
            plt.axis("off")
            plt.show()