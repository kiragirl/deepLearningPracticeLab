from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt


def loaImageAndPredict(imageName, model):
    num_px = 64
    image = Image.open(imageName)
    image_resized = image.resize((num_px, num_px), Image.LANCZOS)
    image_array = np.array(image_resized)
    # # 这会添加一个批次维度，变为(1, 64, 64, 3)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    return predictions


loaded_model = load_model('hand_recognition_model.h5')
print(loaImageAndPredict("0.jpg", loaded_model))
print(loaImageAndPredict("1.jpg", loaded_model))
print(loaImageAndPredict("2.jpg", loaded_model))
print(loaImageAndPredict("3.jpg", loaded_model))
print(loaImageAndPredict("5.jpg", loaded_model))
print(loaImageAndPredict("00.png", loaded_model))
