import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Number of data points:")
print("X_train:", X_train.shape[0])
print("X_temp:", X_temp.shape[0])
print("X_val:", X_val.shape[0])
print("X_test:", X_test.shape[0])

model = Sequential()
model.add(Dense(units=200, activation='relu', input_dim=X_train.shape[1]))  # Hidden layer with 10 units
model.add(Dense(units=200, activation='relu', input_dim=X_train.shape[1]))  # Hidden layer with 10 units
model.add(Dense(units=200, activation='relu', input_dim=X_train.shape[1]))  # Hidden layer with 10 units
model.add(Dense(units=200, activation='relu', input_dim=X_train.shape[1]))  # Hidden layer with 10 units
model.add(Dense(units=3, activation='softmax'))  # Output layer with 3 units (multiclass classification)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()
# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
#
# # Evaluate the model on the testing set
# y_pred = model.predict(X_test)
# y_pred_labels = np.argmax(y_pred, axis=1)
#
# # Compute the confusion matrix
# confusion = confusion_matrix(y_test, y_pred_labels)
# classification_report = classification_report(y_test, y_pred_labels)
#
# # Print evaluation metrics
# print("Confusion Matrix:")
# print(confusion)
# print("Classification Report:")
# print(classification_report)
#
# # Compute ROC curve and AUC for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(len(data.target_names)):
# y_test_binary = np.where(y_test = i, 1, 0)
# y_pred_binary = y_pred[:, i]
# fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_pred_binary)
# roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Plot ROC curve for each class
# plt.figure(figsize=(10, 5))
# for i in range(len(data.target_names)):
# plt.plot(fpr[i], tpr[i], label='ROC curve for class %s (area = %0.2f)' % (data.target_names[i], roc_auc[i]))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0, 1])
# plt.ylim([0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc='lower right')
# plt.show()
#
# # Compute precision-recall curve and average precision for each class
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(len(data.target_names)):
# y_test_binary = np.where(y_test = i, 1, 0)
# y_pred_binary = y_pred[:, i]
# precision[i], recall[i], _ = precision_recall_curve(y_test_binary, y_pred_binary)
# average_precision[i] = auc(recall[i], precision[i])
#
# # Plot precision-recall curve for each class
# plt.figure(figsize=(10, 5))
# for i in range(len(data.target_names)):
# plt.plot(recall[i], precision[i], label='Precision-Recall curve for class %s (AP = %0.2f)' % (data.target_names[i], average_precision[i]))
# plt.xlim([0, 1])
# plt.ylim([0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='lower right')
# plt.show()
#
# # Compute confusion matrix, sensitivity, and specificity for each class
# sensitivity = dict()
# specificity = dict()
# for i in range(len(data.target_names)):
# y_test_binary = np.where(y_test = i, 1, 0)
# y_pred_binary = np.where(y_pred_labels = i, 1, 0)
# tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
# sensitivity[i] = tp / (tp + fn)
# specificity[i] = tn / (tn + fp)
#
# # Print sensitivity and specificity for each class
# for i in range(len(data.target_names)):
# print("Class", data.target_names[i])
# print("Sensitivity:", sensitivity[i])
# print("Specificity:", specificity[i])
# print()
#
# from sklearn.metrics import f1_score
#
# # Compute F1 score for each class
# f1_scores = {}
# for i in range(len(data.target_names)):
# f1_scores[data.target_names[i]] = f1_score(y_test = i, y_pred_labels = i)
#
# # Print F1 score for each class
# for class_name, f1 in f1_scores.items():
# print("Class:", class_name)
# print("F1 Score:", f1)
# print()
#
# # Compute sensitivity and specificity
# true_positive = np.diag(confusion)
# false_positive = np.sum(confusion, axis=0) - true_positive
# false_negative = np.sum(confusion, axis=1) - true_positive
# true_negative = np.sum(confusion) - (true_positive + false_positive + false_negative)
# sensitivity = true_positive / (true_positive + false_negative)
# specificity = true_negative / (true_negative + false_positive)
# # Compute precision for each class
# precision_per_class = true_positive / (true_positive + false_positive)
# # Print evaluation metrics
# print("Confusion Matrix:")
# print(confusion)
# print("Sensitivity:", sensitivity)
# print("Specificity:", specificity)
# print("F1 Score:", f1_score)
# print("Precision per class:", precision_per_class)