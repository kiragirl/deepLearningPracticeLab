from tensorflow.keras.models import load_model
from resnets_utils import *
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

model = load_model('ResNet50.h5')
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


# preds = model.evaluate(X_test, Y_test)
# print("Loss = " + str(preds[0]))
# print("Test Accuracy = " + str(preds[1]))


def predictImage(imageName, model):
    img = image.load_img(imageName, target_size=(64, 64))
    x = image.img_to_array(img) / 255.
    plt.imshow(x)
    plt.axis('off')  # 关闭坐标轴显示，使得图像更加干净
    plt.show()
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    probabilities = model.predict(x)
    print(probabilities)
    predicted_class_index = np.argmax(probabilities, axis=1)

    # 创建独热编码向量
    one_hot_encoding = np.eye(probabilities.shape[1])[predicted_class_index]

    print("预测的类别索引:", predicted_class_index)
    print("独热编码向量:", one_hot_encoding)


imagePath = "13.jpg"
# with Image.open(imagePath) as img:
#     # 显示图片
#     img.show()
predictImage(imagePath, model)
# index = 100
# x = X_test[index]
# print(x.shape)
# plt.imshow(x)
# plt.axis('off')  # 关闭坐标轴显示，使得图像更加干净
# plt.show()
# x = np.expand_dims(x, axis=0)
# print(Y_test[index])
# print(model.predict(np.expand_dims(x, axis=0)))
