#import libraries
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten,concatenate, Reshape, Dense, multiply, Lambda
from keras.layers import ReLU, Add, GlobalAveragePooling2D, Dense,MaxPooling2D, BatchNormalization, Activation, Attention
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

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
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

#Generate the train dataset, validation dataset and test dataset
train_generator=train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator=train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

test_generator=test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

def custom_loss(y_true, y_pred):
    # 避免除以零，添加一个很小的常数
    epsilon = 1e-15
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    # 计算分类交叉熵损失
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-9)) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

    return loss


def create_model1(inputs):
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    x = GlobalAveragePooling2D()(pool2)

    model = Model(inputs=inputs, outputs=x)
    return model


def resnet_block(x, filters, kernel_size=3, stride=1):
    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(y)
    y = BatchNormalization()(y)

    # Adding a shortcut connection
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([y, shortcut])
    else:
        x = Add()([y, x])

    x = Activation('relu')(x)
    return x

def attention_block(x, filters):
    # 注意力机制
    attention = Attention()([x, x])

    # 将注意力应用于输入
    x = Add()([x, attention])
    return x


def create_model2(inputs):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    x = resnet_block(x, 32)
    x = resnet_block(x, 32)

    # 添加注意力机制
    x = attention_block(x, 32)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    # 添加注意力机制
    x = attention_block(x, 64)

    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def create_ensemble_model():
    input_tensor = Input(shape=(224, 224, 3))

    # Create Model1
    model1 = create_model1(input_tensor)

    # Create Model2
    model2 = create_model2(input_tensor)

    # Concatenate the outputs of both models
    merged = concatenate([model1.output, model2.output])

    #     x = GlobalAveragePooling2D()(merged)

    output_tensor = Dense(num_classes, activation='softmax')(merged)

    # Create the ensemble model
    ensemble_model = Model(inputs=input_tensor, outputs=output_tensor)
    return ensemble_model

ensemble_model = create_ensemble_model()
ensemble_model.summary()

ensemble_model.compile(optimizer=Adam(learning_rate=0.01),
              loss=custom_loss,
              metrics=['accuracy'])

history=ensemble_model.fit(train_generator, batch_size=batch_size, epochs=5, validation_data=(validation_generator), verbose=1)

#Training accuracy and validation accuracy graph
plt.figure(figsize=(4,4))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc ='lower right')
plt.show()

#Trainig loss and validation loss graph
plt.figure(figsize=(4,4))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'], loc ='upper right')
plt.show()

#Performance on test set
test_loss, test_acc = ensemble_model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 获取模型的预测结果
y_pred = ensemble_model.predict(test_generator)
y_true = test_generator.classes

# 将概率转换为类别
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
confusion = confusion_matrix(y_true, y_pred_labels)

print("Confusion Matrix:")
print(confusion)

# # 可视化混淆矩阵
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=class_labels)
# disp.plot(cmap='viridis', values_format='d')
# plt.title('Confusion Matrix')
# plt.show()

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
# 清理 GPU 内存
import gc
from numba import cuda

K.clear_session()
gc.collect()

cuda.select_device(0)
cuda.close()

