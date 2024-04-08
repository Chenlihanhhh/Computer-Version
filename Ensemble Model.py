#import libraries

import os

import numpy as np
#import numpy as np
#from keras.applications.vgg16 import VGG16
#from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, AveragePooling2D, \
    GlobalAveragePooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data preparation
dataset_dir= r'D:\桌面\Rice_Image_Dataset'
input_shape=(224, 224, 3)
batch_size=4
num_classes=5

#Data augmentation
train_datagen=ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
validation_split=0.2)

test_datagen=ImageDataGenerator(
rescale=1./255)

#Generate the train dataset, validation dataset and test dataset
train_generator=train_datagen.flow_from_directory(
os.path.join(dataset_dir, 'train'),
target_size=input_shape[:2],
batch_size=batch_size,
class_mode='categorical', #skin has two classes: benign and malignant
subset='training')

validation_generator=train_datagen.flow_from_directory(
os.path.join(dataset_dir, 'train'),
target_size=input_shape[:2],
batch_size=batch_size,
class_mode='categorical', #skin has two classes: benign and malignant
subset='validation')

test_generator=test_datagen.flow_from_directory(
os.path.join(dataset_dir, 'test'),
target_size=input_shape[:2],
batch_size=batch_size,
class_mode='categorical' #skin has two classes: benign and malignant
)

#Ensemble Model

# Define the first CNN network
def create_model1(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    model = Model(inputs=inputs, outputs=pool1)
    return model

# Define the second CNN network
def create_model2(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    model = Model(inputs=inputs, outputs=pool1)
    return model

# Combine the two models into an ensemble model
def create_ensemble_model(input_shape):
    input_layer = Input(shape=input_shape)
    model1 = create_model1(input_shape)(input_layer)
    model2 = create_model2(input_shape)(input_layer)
    merged = concatenate([model1, model2])
    averge_pooling = GlobalAveragePooling2D()(merged)
    output_layer = Dense(num_classes, activation='softmax')(averge_pooling)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Define the input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 5

# Create the ensemble model
ensemble_model = create_ensemble_model(input_shape)

ensemble_model.summary()

# Compile and train the ensemble model
ensemble_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Train the model
history=ensemble_model.fit(train_generator, epochs=1, validation_data=(validation_generator), verbose=1)

#Training accuracy and validation accuracy graph
plt.figure(figsize=(8,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc ='lower right')
plt.show()

#Trainig loss and validation loss graph
plt.figure(figsize=(8,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'], loc ='upper right')
plt.show()

# Evaluate the model on the testing dataset
test_loss, test_acc = ensemble_model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#Compute confusion matrix
from sklearn.metrics import confusion_matrix

# Generate predictions for the test dataset
y_pred = ensemble_model.predict(test_generator)

# Get the predicted labels by selecting the class with the highest probability
y_pred_labels = np.argmax(y_pred, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Compute the confusion matrix
confusion = confusion_matrix(y_true, y_pred_labels)
print("Confusion Matrix:")
print(confusion)


#compute f1-score
from sklearn.metrics import f1_score

f1=f1_score(y_true, y_pred_labels)
print('F1-score:', f1)

#compute precision-recall score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

#compute precision, recall(sensitivity), and threshold values
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

#plot the precision-recall curve
plt.figure(figsize=(8,8))
plt.plot(recall, precision)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

#compute precision-recall score
from sklearn.metrics import roc_curve, auc

#compute precision, recall(sensitivity), and threshold values
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

#compute the AUC score
roc_auc= auc(fpr, tpr)

#plot the precision-recall curve
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, label= 'ROC curve(area =%0.2f)' %roc_auc)
plt.plot([0, 1], [0,1], 'k--')
plt.title('Receiver Operating characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

#compute sensitivity and specificity
sensitivity=tpr[1] #True positive rate
specificity= 1-fpr[1]

print('sensitivity:', sensitivity)
print('specificity:', specificity)

from keras import backend as K
import gc

K.clear_session()
gc.collect()

del ensemble_model



#You need to install numba using 'pip install numba'

from numba import cuda

cuda.select_device(0)
cuda.close()