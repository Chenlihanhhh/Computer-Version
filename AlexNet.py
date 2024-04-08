#import libraries

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data preparation
dataset_dir= r'D:\桌面\crc_skin_data'
input_shape=(224, 224, 3)
batch_size=8
num_classes=2

#Data augmentation
train_datagen=ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

test_datagen=ImageDataGenerator(
rescale=1./255)

#Generate the train dataset, validation dataset and test dataset
train_generator=train_datagen.flow_from_directory(
os.path.join(dataset_dir, 'train'),
target_size=input_shape[:2],
batch_size=batch_size,
class_mode='binary ' #skin has two classes: benign and malignant
)

validation_generator=train_datagen.flow_from_directory(
os.path.join(dataset_dir, 'train'),
target_size=input_shape[:2],
batch_size=batch_size,
class_mode='binary' #skin has two classes: benign and malignant
)

test_generator=test_datagen.flow_from_directory(
os.path.join(dataset_dir, 'test'),
target_size=input_shape[:2],
batch_size=batch_size,
class_mode='binary' #skin has two classes: benign and malignant
)

#building the model: AlexNet: 8 layers - 5 Conv layers and 3 Fully-connected (FC) layers
model=Sequential()
#Add the convolutional layers
model.add(Conv2D(96, kernel_size=(11, 11), strides=(4,4), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

#Flatten the layers
model.add(Flatten())

#Add the fully connected layers
model.add(Dense(4096, activation='relu')) # first FC
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu')) #second FC
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) # third FC

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#Train the model
history=model.fit(train_generator, epochs=10, validation_data=(validation_generator), verbose=1)

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