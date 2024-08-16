from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def predictImage(imageName, model):
    img = image.load_img(imageName, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(model.predict(x))


loaded_model = load_model('ResNet50.h5')
predictImage("0.jpg", loaded_model)
predictImage("1.jpg", loaded_model)
predictImage("2.jpg", loaded_model)
predictImage("3.jpg", loaded_model)