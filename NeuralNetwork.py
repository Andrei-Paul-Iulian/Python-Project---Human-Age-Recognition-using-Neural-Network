import numpy as np
import matplotlib.image as im
import pandas as pd
import cv2
import xlsxwriter
from matplotlib import pyplot as plt
from sklearn import utils
import random

data = pd.read_csv('age.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train = data[0:m].T
Y_train = data_train[0]
for i in range(Y_train.size):
    if(Y_train[i] >= 100):
        Y_train[i] = 10
    elif(Y_train[i] < 10):
        Y_train[i] = 0
    elif(Y_train[i] < 100 and Y_train[i] >= 10):
        Y_train[i] = (Y_train[i] - Y_train[i] % 10)/10
X_train = data_train[1:n]
X_train = X_train / 255.

def init_params():
    W1 = np/random.rand(50, 2304) - 0.5
    b1 = np.random.rand(50, 1) - 0.5
    W2 = np.random.rand(11, 50) - 0.5
    b2 = np.random.rand(11, 1) - 0.5
    return W1, b1, W2, b2

def ReLu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLu_deriv(Z):
    return Z > 0

def matY(Y):
    mat_Y = np.zeros((Y.size, Y.max() + 1))
    mat_Y[np.arange(Y.size), Y] = 1
    mat_Y = mat_Y.T
    return mat_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    mat_Y = matY(Y)
    dZ2 = A2 - mat_Y
    dW2 = 2/m*dZ2.dot(A1.T)
    db2 = 2/m*np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2)
    dW1 = 2/m*dZ1.dot(X.T)
    db1 = 2/m*np.sum(dZ1)
    return dW1, db1, dW2, db2

def incepe(X_train, Y_train, alfa, iteration):
    W1, b1, W2, b2 = init_params()

    cst = 10
    while cst:
        amestec = np.random.permutation(X_train[1,:].size)
        X_train = X_train[:,amestec]
        Y_train = Y_train[amestec]
        W1, b1, W2, b2 = gradient(X_train, Y_train, alfa, iteration, W1, b1, W2, b2)
        cst = cst - 1
    return W1, b1, W2, b2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient(X, Y, alpha, iterations , W1, b1, W2, b2):
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = incepe(X_train, Y_train, 0.10, 10000)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:,index,None]
    prediction = make_predictions(X_train[:,index,None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction * 10, "-", (prediction + 1)*10)
    print("Label: ", label * 10, "-",(label + 1) * 10)

    current_image = current_image.reshape((48, 48)) *255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)