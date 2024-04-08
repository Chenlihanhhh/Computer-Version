import os
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from keras.layers import GlobalAveragePooling2D, Dense, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


input_shape = (224, 224, 3)
dataset_dir = r'D:\桌面\crc_skin_data\B0101T.gdf'
batch_size = 4

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary')



# Define the residual block
def residual_block(inputs, num_filter):
    # Convolution layers
    x = Conv2D(num_filter, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filter, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # Skip connection
    if inputs.shape[-1] != num_filter:
        shortcut = Conv2D(num_filter, kernel_size=(1, 1), padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs
        # add skip connection
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
    return x



# Define the ResNet Model
def ResNet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Convolution layer
    x = Conv2D(64, kernel_size=(7, 7), padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, 3)(x)
    x = Activation('relu')(x)

    # Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Global Average pooling and the Fully connected layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)

    # output layer
    outputs = Dense(num_classes, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model



# Create The ResNet
num_classes = 1
model = ResNet(input_shape, num_classes)

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
history = model.fit(train_generator, epochs=5, validation_data=validation_generator, verbose=1)

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

test_loss, test_accuracy = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

from keras import backend as K
import gc

K.clear_session()
gc.collect()

del model

# you will need to install numba using "pip install numba"
from numba import cuda

cuda.select_device(0)
cuda.close()