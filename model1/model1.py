from utils.data_preparation import X_train, y_train, X_test, y_test
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np 
import itertools

label_map = ['angry', 'sad', 'surprised', 'happy']
num_class = 4

def model1(num_class): #sigmoid na koncu, dropout 0.2, 4 conv, 2fc
    # Initialising the CNN
    model = Sequential()

    model.add(Conv2D(64,(3,3), border_mode='same', input_shape=(144, 144,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(5,5), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,(3,3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,(3,3), border_mode='same'))
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

def plot_results(history):
    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam | Model 1', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('model1_history.png')

def show_classification_report(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    print(classification_report(y_test, y_pred, target_names = label_map))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix | Model 1',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
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

def show_confusion_matrix(y_test, y_pred, label_map):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cnf_matrix, classes=label_map, normalize=True, title='Confusion matrix, with normalization | Model 1')
    plt.savefig('confusion1.png')

def train(model, X_train, y_train, X_test, y_test):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint('best_model1.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    history = model.fit(X_train, y_train,
                batch_size=128,
                epochs=50,
                verbose=2,
                validation_data=(X_test, y_test),
                callbacks=[es, mc])

    model_json = model.to_json()
    with open("model_1.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model1_weigths.h5")
    print("Saved model to disk")

    model.save('model_1.h5')
    plot_results(history)
    print(model.summary())
    pred = show_classification_report(model, X_test, y_test)
    show_confusion_matrix(y_test, pred, label_map)    

model1 = model1(num_class)
train(model1, X_train, y_train, X_test, y_test)