
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from nst2 import *
# 示例图像路径
content_image_path = 'louvre.jpg'
style_image_path = 'monet.jpg'

# 运行风格迁移
generated_image = style_transfer(content_image_path, style_image_path, checkpoint_dir='./checkpoints')

# 显示结果
plt.imshow(generated_image)
plt.title("Generated Image")
plt.axis('off')
plt.show()