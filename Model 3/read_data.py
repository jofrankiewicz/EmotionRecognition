import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

with open('history3.json') as f:
    d = json.load(f)
    print(d)

d.keys()

categorical_accuracy = d['acc']
loss = d['loss']
val_categorical_accuracy = d['val_acc']
val_loss = d['val_loss']

categorical_accuracy_df = categorical_accuracy.values()
categorical_accuracy_list = list(categorical_accuracy_df)

loss_df = loss.values()
loss_list = list(loss_df)

val_categorical_accuracy_df = val_categorical_accuracy.values()
val_categorical_accuracy_list = list(val_categorical_accuracy_df)

val_loss_df = val_loss.values()
val_loss_list = list(val_loss_df)

len(val_loss_list)

history = pd.DataFrame(list(zip(categorical_accuracy_list, loss_list, val_categorical_accuracy_list, val_loss_list)), columns =['categorical_accuracy', 'loss', 'val_categorical_accuracy', 'val_loss'])
history.info()

def plot_results(history):
    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam | Model 3', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()

plot_results(history)

y_test = pd.read_csv('test_vector_3.csv')
y_predicted = pd.read_csv('predicted_vector_3.csv')

y_test['result'] = y_test['Zbior testowy'].apply(lambda x: x.index('1'))
y_test.result = y_test.result.apply(lambda x: 0 if x == 1 else x)
y_test.result = y_test.result.apply(lambda x: 1 if x == 4 else x)
y_test.result = y_test.result.apply(lambda x: 2 if x == 7 else x)
y_test.result = y_test.result.apply(lambda x: 3 if x == 10 else x)

list_of_predictions = []
for el in range(len(y_predicted)):
    y_predicted_list = np.array(y_predicted.iloc[el])
    y_el = y_predicted_list[0].split()
    list_of_predictions.append(y_el)

y_predicted['list_of_values'] = list_of_predictions

first_element = []
for el in range(len(y_predicted)):
    element = y_predicted.list_of_values[el][0]
    first_element.append(element)

second_element = []
for el in range(len(y_predicted)):
    element = y_predicted.list_of_values[el][1]
    second_element.append(element)

third_element = []
for el in range(len(y_predicted)):
    element = y_predicted.list_of_values[el][2]
    third_element.append(element)

fourth_element = []
for el in range(len(y_predicted)):
    element = y_predicted.list_of_values[el][3]
    fourth_element.append(element)

predictions = pd.DataFrame(list(zip(first_element, second_element, third_element, fourth_element)), columns =['0', '1', '2', '3'])
predictions['0'] = predictions['0'].apply(lambda x: x.replace('[', ''))
predictions['3'] = predictions['3'].apply(lambda x: x.replace(']', ''))

predictions['0'] = predictions['0'].astype(float)
predictions['1'] = predictions['1'].astype(float)
predictions['2'] = predictions['2'].astype(float)
predictions['3'] = predictions['3'].astype(float)

predictions.info()
predictions['max_element'] = predictions.idxmax(axis=1)
predictions.max_element = predictions.max_element.astype(int)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix | Model 3',
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
    plot_confusion_matrix(cnf_matrix, classes=label_map, normalize=True, title='Confusion matrix, with normalization | Model 3')
    plt.show()

label_map = ['angry', 'sad', 'surprised', 'happy']
show_confusion_matrix(y_test.result,  predictions.max_element, label_map)