import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
# from tensorflow.python.framework import ops
from cnn_utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import os

print("当前工作目录:", os.getcwd())
print(os.path.abspath('datasets/train_signs.h5'))
if os.path.exists('datasets/train_signs.h5'):
    print("文件存在")
else:
    print("文件不存在")
# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Flatten the training and test images
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
# X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
# X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# # Normalize image vectors
# X_train = X_train_flatten / 255.
# X_test = X_test_flatten / 255.
# # Convert training and test labels to one hot matrices
# Y_train = convert_to_one_hot(Y_train_orig, 6)
# Y_test = convert_to_one_hot(Y_test_orig, 6)
# X_train_corrected = X_train.T
# Y_train_corrected = Y_train.T

model = Sequential()
optimizer = tf.optimizers.Adam(learning_rate=0.0009)

# model.add(layers.Conv2D(8, (4, 4), strides=(1, 1), padding='SAME', activation='relu', input_shape=(64, 64, 3)))
# model.add(layers.MaxPooling2D((8, 8), strides=(8, 8), padding='SAME'))
# model.add(layers.Conv2D(16, (2, 2), strides=(1, 1), padding='SAME', activation='relu'))
# model.add(layers.MaxPooling2D((4, 4), strides=(4, 4), padding='SAME'))
model.add(layers.Conv2D(8, (4, 4), padding='SAME', activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((8, 8), padding='SAME'))
model.add(layers.Conv2D(16, (2, 2), padding='SAME', activation='relu'))
model.add(layers.MaxPooling2D((4, 4), padding='SAME'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 添加全连接层
model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))\
# 不指定激活函数为softmax，预测时不会输出one hot编码
model.add(layers.Dense(6, activation='softmax'))
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=400)

train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

model.save('hand_recognition_model.h5')

print("Train Accuracy:", train_acc)
print("Train loss:", train_loss)
print("Test Accuracy:", test_acc)
print("Test loss:", test_loss)

# 提取历史记录中的训练损失和验证损失
epochs = range(1, len(history.history['loss']) + 1)
train_losses = history.history['loss']
# val_losses = history.history['val_loss']

# 绘制损失曲线
plt.figure()
plt.plot(epochs, train_losses, label='Training Loss')
# plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
