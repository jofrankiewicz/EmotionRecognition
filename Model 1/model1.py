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
import tensorflow as tf
import pickle
import keras 
from keras.utils import plot_model

import matplotlib
matplotlib.use('Agg')

config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

filename = '/home/jfrankiewicz/scripts/images_all.csv'
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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

label_map = ['angry', 'sad', 'surprised', 'happy']
num_class = 4

def model1(num_class): #sigmoid na koncu, dropout 0.2, 4 conv, 2fc
    # Initialising the CNN
    model = Sequential()

    model.add(Conv2D(64,kernel_size=3, border_mode='same', input_shape=(144, 144,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,kernel_size=5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,kernel_size=3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,kernel_size=3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(num_class, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix | Model 1'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure()
    plt.plot(cm, interpolation='nearest')
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig('confusionmatrix1.png') 

def show_confusion_matrix(y_test, y_pred, label_map):
    y_test = list(y_test)
    y_pred = list(y_pred)
    y_actu = pd.Series(y_test, name='Zbior testowy')
    y_pred = pd.Series(y_pred, name='Przewidywanie')
    y_actu.to_csv('test_vector_1.csv',  header=True, index=False)
    y_pred.to_csv('predicted_vector_1.csv',  header=True, index=False)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #plot_confusion_matrix2(cnf_matrix, target_names=label_map)

def train(model, X_train, y_train, X_test, y_test):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint('best_model1.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    history = model.fit(X_train, y_train,
                batch_size=128,
                epochs=100,
                verbose=2,
                validation_data=(X_test, y_test),
                callbacks=[es, mc])

                 
    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = 'history1.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    model_json = model.to_json()
    with open("model_1.json", "w") as json_file:
        json_file.write(model_json)

    plot_model(model, to_file='model1.png')

    print(model.summary())
    y_pred = model.predict(X_test)
    show_confusion_matrix(y_test, y_pred, label_map)  

model1 = model1(num_class)
train(model1, X_train, y_train, X_test, y_test)