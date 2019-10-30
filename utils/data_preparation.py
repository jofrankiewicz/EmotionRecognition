import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

filename = '../images_all.csv'
label_map = ['angry', 'sad', 'surprised', 'happy']

def getData(filename):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[4]))
            X.append([int(p) for p in row[2].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

def balance_class(Y):
    num_class = set(Y)
    count_class = {}
    for i in range(len(num_class)):
        count_class[i] = sum([1 for y in Y if y == i])
    return count_class

X, Y = getData(filename)
num_class = len(set(Y))
balance = balance_class(Y)
N, D = X.shape
X = X.reshape(N, 144, 144, 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)


