import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import os
print("当前工作目录:", os.getcwd())
print(os.path.abspath('train_signs.h5'))
if os.path.exists('train_signs.h5'):
    print("文件存在")
else:
    print("文件不存在")
# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
X_train_corrected = X_train.T
Y_train_corrected = Y_train.T

model = Sequential()
optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    # 添加全连接层
model.add(Dense(25, activation='relu', input_dim=12288))  # 第一个隐藏层，假设输入维度是128
model.add(Dense(12, activation='relu'))  # 第二个隐藏层
model.add(Dense(6, activation='softmax'))
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    # train_dataset = train_dataset.shuffle(buffer_size=len(Y_train)).batch(minibatch_size)
history = model.fit(X_train_corrected, Y_train_corrected, epochs=1500)

train_loss, train_acc = model.evaluate(X_train_corrected, Y_train_corrected, verbose=2)
test_loss, test_acc = model.evaluate(X_test.T, Y_test.T, verbose=2)

model.save('my_model.h5')

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
