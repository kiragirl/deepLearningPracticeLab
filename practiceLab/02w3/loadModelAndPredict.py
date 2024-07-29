from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt

def loaImageAndPredict(imageName):
    num_px = 64
    image = Image.open(imageName)
    plt.imshow(image)
    image_resized = image.resize((num_px, num_px), Image.LANCZOS)
    image_array = np.array(image_resized)
    my_image = image_array.reshape((1, num_px*num_px*3)).T

    # 加载 HDF5 格式的模型
    loaded_model = load_model('my_model.h5')

    # 现在可以使用 loaded_model 进行预测
    predictions = loaded_model.predict(my_image.T)
    return predictions
print(loaImageAndPredict("12.jpg"))
print(loaImageAndPredict("13.jpg"))
print(loaImageAndPredict("14.jpg"))
print(loaImageAndPredict("15.jpg"))
print(loaImageAndPredict("11.jpg"))