from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def loaImageAndPredict(imageName, model):
    num_px = 64
    image = Image.open(imageName)
    image_resized = image.resize((num_px, num_px), Image.LANCZOS)
    image_array = np.array(image_resized)
    # # 这会添加一个批次维度，变为(1, 64, 64, 3)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    return predictions


def predictImage(imageName, model):
    img = image.load_img(imageName, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(model.predict(x))


loaded_model = load_model('happy_house.keras')
predictImage("my_image.jpg", loaded_model)
