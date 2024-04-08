import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize
# from keras_flops import get_flops

# Data preparation
# dataset_dir= r'E:\2023 OBU-CDUT\2023 Semester 1\2023 Research - Hace\Plant disease'
dataset_dir = r'/mnt/Dataset/colored_images'
input_shape = (224, 224, 3)
batch_size = 4
num_classes=5

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

test_datagen = ImageDataGenerator(
    rescale=1. / 255)

# Generate the train dataset, validation dataset and test dataset
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',  # plant disease of two classes:healthy and infected
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',  # plant disease of two classes:healthy and infected
    subset='validation')

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'  # plant disease of two classes:healthy and infected
)



# callback
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def DepthwiseConv():
    model = Sequential()

    model.add(DepthwiseConv2D((3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(512, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.35))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(DepthwiseConv2D((3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.35))

    model.add(DepthwiseConv2D((3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(DepthwiseConv2D((3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.35))

    model.add(DepthwiseConv2D((3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(5, activation='softmax'))
    return model


def SeparableConv():
    model = Sequential()

    model.add(SeparableConv2D(512, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.35))
    model.add(SeparableConv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(SeparableConv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.35))
    model.add(SeparableConv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(SeparableConv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.35))
    model.add(SeparableConv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    # model.add(Dense(64, activation='relu'))

    model.add(Dense(5, activation='softmax'))
    # model.add(Dense(num_classes, activation='sigmoid'))

    return model


def DepthwiseSeparableConv():
    model = Sequential()

    model.add(DepthwiseConv2D((3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(512, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.35))
    model.add(SeparableConv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(DepthwiseConv2D((3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.35))
    model.add(SeparableConv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(DepthwiseConv2D((3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.35))
    model.add(SeparableConv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(5, activation='softmax'))
    return model


Model1 = DepthwiseConv()
Model2 = SeparableConv()
Model3 = DepthwiseSeparableConv()

models = [Model1, Model2, Model3]
times = []
historys = []

for model in models:
    time_callback = TimeHistory()
    times.append(time_callback)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, batch_size=batch_size, epochs=1,
                        validation_data=validation_generator, verbose=1, callbacks=[time_callback])

    historys.append(history)

# Loss
modellist = ["DepthwiseConv", "SeparableConv", "DepthwiseSeparableConv"]
color = ['#1f77b4', '#ff7f0e', '#2ba02b']

for index in range(len(historys)):
    plt.plot(historys[index].history['val_loss'], label=modellist[index], color=color[index])

plt.ylabel('Validation Loss')
plt.xlabel('Epoch')
plt.legend(loc=0)
plt.show()

# Accuracy
for index in range(len(historys)):
    plt.plot(historys[index].history['val_accuracy'], label=modellist[index], color=color[index])

plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(loc=0)
plt.show()

# Performance Evaluation
for index, model in enumerate(models):
    print(f"Model {modellist[index]}:")

    # Evaluate the model on the testing dataset
    test_loss, test_acc = model.evaluate(test_generator)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Generate predictions for the test dataset
    y_pred = model.predict(test_generator)

    # Get the true labels from the test generator
    y_true = test_generator.classes

    # Convert true labels to one-hot encoded format
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Compute confusion matrix
    confusion = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
    print("Confusion Matrix:")
    print(confusion)

    # Compute F1-score
    f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted')  # Use 'weighted' for multi-class
    print('F1-score:', f1)

    # Compute precision-recall curve
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    # Compute micro-average precision-recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(),
        y_pred.ravel()
    )
    average_precision["micro"] = auc(
        recall["micro"],
        precision["micro"]
    )

    # Plot the precision-recall curve for each class
    plt.figure(figsize=(8, 8))
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
             label='micro-average (area = {0:0.2f})'.format(average_precision["micro"]))
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], lw=2, label='class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Compute ROC curve and AUC score
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC score
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot the ROC curve for each class
    plt.figure(figsize=(8, 8))
    plt.plot(fpr["micro"], tpr["micro"], color='gold', lw=2,
             label='micro-average (area = {0:0.2f})'.format(roc_auc["micro"]))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

from keras import backend as K
import gc

K.clear_session()
gc.collect()

del model

# you will need to install numba using "pip install numba"
from numba import cuda

cuda.select_device(0)
cuda.close()