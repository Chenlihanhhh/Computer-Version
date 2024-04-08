from sklearn.metrics import confusion_matrix, classification_report,precision_score,recall_score,roc_auc_scorce,classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from keras.layers import *
from keras.layers.normalization import BatchNormalization
import numpy.random as rng
#install pip pydot_ng
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Lambda,Flatten,Dropout,Activation,AveragePooling2D,BatchNormalization
from keras.layers.merge import add, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import pydot_ng as pydot

#install pip pydot_ng

print (pydot.find_graphviz())
np.random.seed(42)
plt.rcParams['image.cmap'] = 'gray'


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]



# batch operation usng tensor slice
def WaveletTransformAxisY(batch_img):
    odd_img = batch_img[:, 0::2]
    even_img = batch_img[:, 1::2]
    L = (odd_img + even_img) / 2.0
    H = K.abs(odd_img - even_img)
    return L, H


def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = K.permute_dimensions(batch_img, [0, 2, 1])[:, :, ::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = K.permute_dimensions(_dst_L, [0, 2, 1])[:, ::-1, ...]
    dst_H = K.permute_dimensions(_dst_H, [0, 2, 1])[:, ::-1, ...]
    return dst_L, dst_H


def Wavelet(batch_image):
    # make channel first image
    batch_image = K.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:, 0]
    g = batch_image[:, 1]
    b = batch_image[:, 2]

    # level 1 decomposition
    wavelet_L, wavelet_H = WaveletTransformAxisY(r)
    r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(g)
    g_wavelet_LL, g_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    g_wavelet_HL, g_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(b)
    b_wavelet_LL, b_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    b_wavelet_HL, b_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH,
                    g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                    b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
    transform_batch = K.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(g_wavelet_LL)
    g_wavelet_LL2, g_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    g_wavelet_HL2, g_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(b_wavelet_LL)
    b_wavelet_LL2, b_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    b_wavelet_HL2, b_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2,
                       g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                       b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
    transform_batch_l2 = K.stack(wavelet_data_l2, axis=1)

    # level 3 decomposition
    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(r_wavelet_LL2)
    r_wavelet_LL3, r_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    r_wavelet_HL3, r_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(g_wavelet_LL2)
    g_wavelet_LL3, g_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    g_wavelet_HL3, g_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL2)
    b_wavelet_LL3, b_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    b_wavelet_HL3, b_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3,
                       g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                       b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
    transform_batch_l3 = K.stack(wavelet_data_l3, axis=1)

    # level 4 decomposition
    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(r_wavelet_LL3)
    r_wavelet_LL4, r_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    r_wavelet_HL4, r_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(g_wavelet_LL3)
    g_wavelet_LL4, g_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    g_wavelet_HL4, g_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL3)
    b_wavelet_LL4, b_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    b_wavelet_HL4, b_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4,
                       g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                       b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
    transform_batch_l4 = K.stack(wavelet_data_l4, axis=1)


    decom_level_1 = K.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decom_level_2 = K.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
    decom_level_3 = K.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
    decom_level_4 = K.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])

    return [decom_level_1, decom_level_2, decom_level_3, decom_level_4]


def Wavelet_out_shape(input_shapes):
    # print('in to shape')
    return [tuple([None, 112, 112, 12]), tuple([None, 56, 56, 12]),
            tuple([None, 28, 28, 12]), tuple([None, 14, 14, 12])]


img_batch = K.zeros(shape=(8, 224, 224, 3), dtype='float32')
Wavelet(img_batch)

def get_wavelet_cnn_model():

    input_shape = 224, 224, 3

    input_ = Input(input_shape, name='the_input')
    # wavelet = Lambda(Wavelet, name='wavelet')
    wavelet = Lambda(Wavelet, Wavelet_out_shape, name='wavelet')
    input_l1, input_l2, input_l3, input_l4 = wavelet(input_)
    conv_1 = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_1')(input_l1)
    norm_1 = BatchNormalization(name='norm_1')(conv_1)
    relu_1 = Activation('relu', name='relu_1')(norm_1)

    conv_1_2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_1_2')(relu_1)
    norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
    relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

    # level two decomposition starts
    conv_a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_a')(input_l2)
    norm_a = BatchNormalization(name='norm_a')(conv_a)
    relu_a = Activation('relu', name='relu_a')(norm_a)

    # concate level one and level two decomposition
    concate_level_2 = concatenate([relu_1_2, relu_a])
    conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_2')(concate_level_2)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    relu_2 = Activation('relu', name='relu_2')(norm_2)

    conv_2_2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_2_2')(relu_2)
    norm_2_2 = BatchNormalization(name='norm_2_2')(conv_2_2)
    relu_2_2 = Activation('relu', name='relu_2_2')(norm_2_2)

    # level three decomposition starts
    conv_b = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_b')(input_l3)
    norm_b = BatchNormalization(name='norm_b')(conv_b)
    relu_b = Activation('relu', name='relu_b')(norm_b)

    conv_b_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_b_2')(relu_b)
    norm_b_2 = BatchNormalization(name='norm_b_2')(conv_b_2)
    relu_b_2 = Activation('relu', name='relu_b_2')(norm_b_2)

    # concate level two and level three decomposition
    concate_level_3 = concatenate([relu_2_2, relu_b_2])
    conv_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_3')(concate_level_3)
    norm_3 = BatchNormalization(name='nomr_3')(conv_3)
    relu_3 = Activation('relu', name='relu_3')(norm_3)

    conv_3_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_3_2')(relu_3)
    norm_3_2 = BatchNormalization(name='norm_3_2')(conv_3_2)
    relu_3_2 = Activation('relu', name='relu_3_2')(norm_3_2)

    # level four decomposition start
    conv_c = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_c')(input_l4)
    norm_c = BatchNormalization(name='norm_c')(conv_c)
    relu_c = Activation('relu', name='relu_c')(norm_c)

    conv_c_2 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_2')(relu_c)
    norm_c_2 = BatchNormalization(name='norm_c_2')(conv_c_2)
    relu_c_2 = Activation('relu', name='relu_c_2')(norm_c_2)

    conv_c_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_3')(relu_c_2)
    norm_c_3 = BatchNormalization(name='norm_c_3')(conv_c_3)
    relu_c_3 = Activation('relu', name='relu_c_3')(norm_c_3)

    # concate level level three and level four decomposition
    concate_level_4 = concatenate([relu_3_2, relu_c_3])
    conv_4 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_4')(concate_level_4)
    norm_4 = BatchNormalization(name='norm_4')(conv_4)
    relu_4 = Activation('relu', name='relu_4')(norm_4)

    conv_4_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_4_2')(relu_4)
    norm_4_2 = BatchNormalization(name='norm_4_2')(conv_4_2)
    relu_4_2 = Activation('relu', name='relu_4_2')(norm_4_2)

    conv_5_1 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_5_1')(relu_4_2)
    norm_5_1 = BatchNormalization(name='norm_5_1')(conv_5_1)
    relu_5_1 = Activation('relu', name='relu_5_1')(norm_5_1)

    pool_5_1 = AveragePooling2D(pool_size=(7, 7), strides=1, padding='same', name='avg_pool_5_1')(relu_5_1)
    flat_5_1 = Flatten(name='flat_5_1')(pool_5_1)

    fc_5 = Dense(2048, name='fc_5')(flat_5_1)
    norm_5 = BatchNormalization(name='norm_5')(fc_5)
    relu_5 = Activation('relu', name='relu_5')(norm_5)
    drop_5 = Dropout(0.5, name='drop_5')(relu_5)

    fc_6 = Dense(2048, name='fc_6')(drop_5)
    norm_6 = BatchNormalization(name='norm_6')(fc_6)
    relu_6 = Activation('relu', name='relu_6')(norm_6)
    drop_6 = Dropout(0.5, name='drop_6')(relu_6)

    output = Dense(5, activation='softmax', name='fc_7')(drop_6)

    model = Model(inputs=input_, outputs=output)
    model.summary()
    plot_model(model, to_file='neurowavel_classification.png')


    return model

model = get_wavelet_cnn_model()
plot_model(model, show_shapes=True, show_layer_names=True)

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


import os
data_set_path = r'/mnt/Dataset/colored_images'
os.listdir(data_set_path)


# prepare data augmentation configuration
# Data Division
nb_train_samples = 400
nb_validation_samples = 80
batch_size = 32
# dimensions of our images.
img_width, img_height = 224, 224
input_shape = (224, 224, 3)

img_rows, img_cols, img_channel = 224, 224, 3

img_width, img_height = 224, 224

data_generator = ImageDataGenerator(
    # rescale=1. / 255,
    validation_split=0.2)
# shear_range=0.1,
# zoom_range=0.1,
# horizontal_flip=True)

train_data_dir = r'/mnt/Dataset/colored_images'

train_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size, shuffle=True, seed=13,
    subset="training",
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size, shuffle=False, seed=13,
    subset="validation",
    class_mode='categorical')


METRICS = [
    tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="AUC"),
    tf.keras.metrics.TruePositives(name="true_positives"),
    tf.keras.metrics.TrueNegatives(name="true_negatives"),
    tf.keras.metrics.FalsePositives(name="false_positives"),
    tf.keras.metrics.FalseNegatives(name="false_negatives"),
    keras.metrics.SensitivityAtSpecificity(0.5, name="sensitivity"),
    keras.metrics.SpecificityAtSensitivity(0.5, name="specificity")]

model.compile(optimizer=keras.optimizers.Adam(0.0004), loss="categorical_crossentropy", metrics=METRICS)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=3, verbose=True,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples)

model.save('neurowavel_classification.h5')

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
epochs = range(len(acc))

plt.plot(acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title('Accuracy curve')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title('Loss curve')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

fig, ax = plt.subplots(6, 2, figsize=(15, 30))
ax = ax.ravel()

for i, met in enumerate(["precision", "recall", "binary_accuracy", "loss", "AUC", "true_positives", "true_negatives",
                         "false_positives", "false_negatives", "sensitivity", "specificity"]):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history["val_" + met])
    ax[i].set_title("Model {}".format(met))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(met)
    ax[i].legend(["train", "val"])

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
pre = history.history['precision']
val_pre = history.history['val_precision']
rcall = history.history['recall']
val_rcall = history.history['val_recall']
aruc = history.history['AUC']
val_aruc = history.history['val_AUC']
loss = history.history['loss']
val_loss = history.history['val_loss']
true_pos = history.history['true_positives']
true_neg = history.history['true_negatives']
false_pos = history.history['false_positives']
false_neg = history.history['false_negatives']
val_true_pos = history.history['val_true_positives']
val_true_neg = history.history['val_true_negatives']
val_false_pos = history.history['val_false_positives']
val_false_neg = history.history['val_false_negatives']
sensitivity = history.history['sensitivity']
specificity = history.history['specificity']
val_sensitivity = history.history['val_sensitivity']
val_specificity = history.history['val_specificity']

epochs = range(1, len(acc) + 1)


# Train and validation accuracy
plt.plot(epochs, acc, 'b--', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r-v', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
# Train and validation loss
plt.plot(epochs, loss, 'b--', label='Training loss')
plt.plot(epochs, val_loss, 'r-v', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# Train and validation accuracy on Model Precision
plt.plot(epochs, pre, 'b--', label='Training precision')
plt.plot(epochs, val_pre, 'r-v', label='Validation precision')
plt.title('Model Precision')
plt.legend()


# Train and validation accuracy on Model Recall
plt.plot(epochs, rcall, 'b--', label='Training recall')
plt.plot(epochs, val_rcall, 'r-v', label='Validation recall')
plt.title('Model Recall')
plt.legend()


# Train and validation accuracy on Model AUC
plt.plot(epochs, aruc, 'b--', label='Training AUC')
plt.plot(epochs, val_aruc, 'r-v', label='Validation AUC')
plt.title('AUC')
plt.legend()

# Train and validation accuracy on True positives
plt.plot(epochs, true_pos, 'b--', label='Training True Positives')
plt.plot(epochs, val_true_pos, 'r-v', label='Validation True Positives')
plt.title('Training and Validation True Positives')
plt.legend()
plt.show()


##Train and validation accuracy on True Negatives
plt.plot(epochs, true_neg, 'b--', label='Training True Negatives')
plt.plot(epochs, val_true_neg, 'r-v', label='Validation True Negatives')
plt.title('Training and Validation True Negatives')
plt.legend()
plt.show()


##Train and validation accuracy on False Positives
plt.plot(epochs, false_pos, 'b--', label='Training False Positives')
plt.plot(epochs, val_false_pos, 'r-v', label='Validation False Positives')
plt.title('Training and Validation False Positives')
plt.legend()
plt.show()


##Train and validation accuracy on False Negatives
plt.plot(epochs, false_neg, 'b--', label='Training False Negatives')
plt.plot(epochs, val_false_neg, 'r-v', label='Validation False Negatives')
plt.title('Training and Validation False Negatives')
plt.legend()
plt.show()

##Train and validation accuracy on specificity
plt.plot(epochs, specificity, 'b--', label='Training Specificity')
plt.plot(epochs, val_specificity, 'r-v', label='Validation Specificity')
plt.title('Training and Validation Specificity')
plt.legend()
plt.show()

##Train and validation accuracy on Sensitivity
plt.plot(epochs, sensitivity, 'b--', label='Training Sensitivity')
plt.plot(epochs, val_sensitivity, 'r-v', label='Validation Sensitivity')
plt.title('Training and Validation Sensitivity')
plt.legend()
plt.show()

metrics = model.evaluate(train_generator)
print('Loss of {} and Accuracy is {} %'.format(metrics[0], metrics[1] * 100))

metrics = model.evaluate(validation_generator)
print('Loss of {} and Accuracy is {} %'.format(metrics[0], metrics[1] * 100))

Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size)
y_pred = np.argmax(Y_pred, axis=1)


import itertools

# confusion Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Covid', 'Normal']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sn

warnings.filterwarnings("ignore")
fig = plt.figure(num=None, figsize=(10, 10), dpi=40, facecolor='w', edgecolor='k')
cm = confusion_matrix(validation_generator.classes, y_pred)
conf_matrix = pd.DataFrame(data=cm,
                           columns=['Covid', 'Normal'],
                           index=['Covid', 'Normal'])
sn.set(font_scale=1.8)  # for label size
sn.heatmap(conf_matrix, annot=True, fmt='.0f', annot_kws={"size": 25}, cmap="Oranges")  # font size
plt.title('Confusion Matrix', fontsize=30)
plt.show()

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sn

warnings.filterwarnings("ignore")
fig = plt.figure(num=None, figsize=(10, 10), dpi=40, facecolor='w', edgecolor='k')
cm = confusion_matrix(validation_generator.classes, y_pred)
conf_matrix = pd.DataFrame(data=cm,
                           columns=['Covid', 'Normal'],
                           index=['Covid', 'Normal'])
sn.set(font_scale=1.8)  # for label size
sn.heatmap(conf_matrix / np.sum(conf_matrix), annot=True, fmt='.2%', annot_kws={"size": 25}, cmap="Blues")  # font size
plt.title('Confusion Matrix', fontsize=30)
plt.show()
# conf_matrix

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sn

warnings.filterwarnings("ignore")
fig = plt.figure(num=None, figsize=(10, 10), dpi=40, facecolor='w', edgecolor='k')
cm = confusion_matrix(validation_generator.classes, y_pred)

conf_matrix = pd.DataFrame(data=cm,
                           columns=['Covid', 'Normal'],
                           index=['Covid', 'Normal'])

labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
labels = np.asarray(labels).reshape(2, 2)

sn.set(font_scale=1.8)  # for label size
sn.heatmap(conf_matrix, annot=labels, fmt='', annot_kws={"size": 25}, cmap="Oranges")  # font size
plt.title('Confusion Matrix', fontsize=30)
plt.show()


from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sn

from cf_matrix import make_confusion_matrix

sn.set_context('talk')

warnings.filterwarnings("ignore")
fig = plt.figure(num=None, figsize=(10, 10), dpi=40, facecolor='w', edgecolor='k')
cm = confusion_matrix(validation_generator.classes, y_pred)

conf_matrix = np.array(conf_matrix)

labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Covid', 'Normal']

make_confusion_matrix(conf_matrix, group_names=labels, categories=categories, cmap='Blues', title='Confusion Matrix')




# r = np.flip(sklearn.metrics.confusion_matrix(y_true, y_pred))
r = (confusion_matrix(validation_generator.classes, y_pred))
print("Confusion Matrix")
print(r)

# precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
precision = precision_score(y_true=validation_generator.classes, y_pred=y_pred)
print('precision')
print(precision)

# recall = sklearn.metrics.recall_score(y_true=test_Y, y_pred=y_pred, pos_label="positive")
recall = recall_score(y_true=validation_generator.classes, y_pred=y_pred)
print('recall')
print(recall)

# recall = sklearn.metrics.recall_score(y_true=test_Y, y_pred=y_pred, pos_label="positive")
roc_auc = roc_auc_score(validation_generator.classes, y_pred, average=None)
print('ROC AUC')
print(roc_auc)


# cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
cm = confusion_matrix(validation_generator.classes, y_pred)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[0, 1] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print('Confusion Matrix')
print(cm)
print("accuracy: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))



target_names = ['Covid', 'Normal']

print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


from keras import backend as K
import gc

K.clear_session()
gc.collect()

del model

# you will need to install numba using "pip install numba"
from numba import cuda

cuda.select_device(0)
cuda.close()