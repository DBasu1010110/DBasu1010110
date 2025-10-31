# digit_recognizer.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from PIL import Image
import os

import warnings
warnings.filterwarnings("ignore")

# Module 1- Training Function

def train_model(save_path="digit_model.pkl"):
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print("Training model...")
    model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=20, verbose=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    dump(model, save_path)
    print(f"Model saved to {save_path}")


# Module 2 - Prediction function

def predict_image(image_path, model_path="digit_model.pkl"):
    if not os.path.exists(model_path):
        print("Model not found. Train it first using train_model().")
        return

    model = load(model_path)

    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # invert if white background
    img_array = img_array / 255.0
    img_flatten = img_array.flatten().reshape(1, -1)

    prediction = model.predict(img_flatten)[0]

    plt.imshow(img_array, cmap='gray')
    plt.title(f"Predicted Digit: {prediction}")
    plt.axis('off')
    plt.show()

    return prediction


# Module 3 - Main script for train/test

if __name__ == "__main__":
    print("Training model, please wait...")
    model = train_model()

# Module 4 - Prediction of user input image (preferably a new cell for re-run purposes)

# Enter image path
img_path = input("\nEnter path to the image you want to predict: ").strip()
result = predict_image(img_path)

if result is not None:
    print(f"\nPredicted digit: {result}") #can remove this output
