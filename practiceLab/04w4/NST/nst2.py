import os

import tensorflow as tf
from keras import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt


# 定义辅助函数
def load_and_process_image(image_path, target_dim=(224, 224)):
    img = load_img(image_path, target_size=target_dim)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 定义特征提取器
def get_feature_extractor(style_layers, content_layers):
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]

    feature_extractor = Model(vgg.input, outputs)
    return feature_extractor


# 定义损失函数
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def style_loss(style_outputs, style_targets):
    S = [gram_matrix(style_output) for style_output in style_outputs]
    loss = tf.add_n([tf.reduce_mean((S[i] - style_targets[i]) ** 2)
                     for i in range(len(style_outputs))])
    return loss / float(len(style_outputs))


def content_loss(content_output, content_target):
    loss = tf.reduce_sum((content_output - content_target) ** 2)
    return loss / tf.cast(tf.size(content_output), tf.float32)


def total_variation_loss(image, tv_weight=1e-6):
    x_diff = image[:, 1:, :, :] - image[:, :-1, :, :]
    y_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
    loss = tf.reduce_sum(tf.abs(x_diff)) + tf.reduce_sum(tf.abs(y_diff))
    return tv_weight * loss


# 定义风格迁移函数
def style_transfer(content_image_path, style_image_path,
                   content_layers=['block5_conv2'],
                   style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
                   content_weight=1e3, style_weight=1e-2, tv_weight=1e-6,
                   iterations=1000, learning_rate=2.0, checkpoint_dir='./checkpoints'):
    content_image = load_and_process_image(content_image_path)
    style_image = load_and_process_image(style_image_path)

    feature_extractor = get_feature_extractor(style_layers, content_layers)

    # Extract features from the content and style images
    all_outputs = feature_extractor(style_image)
    style_outputs = all_outputs[:len(style_layers)]
    style_targets = [gram_matrix(output) for output in style_outputs]

    all_outputs = feature_extractor(content_image)
    content_outputs = all_outputs[len(style_layers):]

    # Initialize the generated image with the content image
    generated_image = tf.Variable(content_image)

    # Set up the optimizer
    opt = tf.optimizers.Adam(learning_rate=learning_rate)

    # 创建 Checkpoint 对象
    checkpoint = tf.train.Checkpoint(optimizer=opt, generated_image=generated_image)

    # 检查是否已有 Checkpoint 文件
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).assert_existing_objects_matched()
        print("Restored from checkpoint.")
    else:
        print("No checkpoint found, initializing from scratch.")

    # 训练循环
    for n in range(iterations):
        with tf.GradientTape() as tape:
            all_outputs = feature_extractor(generated_image)
            style_outputs = all_outputs[:len(style_layers)]
            content_output = all_outputs[len(style_layers)]

            style_loss_value = style_loss(style_outputs, style_targets)
            content_loss_value = content_loss(content_output, content_outputs)
            tv_loss_value = total_variation_loss(generated_image, tv_weight)

            loss = content_weight * content_loss_value + style_weight * style_loss_value + tv_loss_value

        grad = tape.gradient(loss, generated_image)
        opt.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, -127.5, 127.5))

        if n % 100 == 0:
            print(
                f"Iteration {n}: Total Loss: {loss:.4f} Content Loss: {content_loss_value:.4f} Style Loss: {style_loss_value:.4f} TV Loss: {tv_loss_value:.4f}")

        # 每隔一段时间保存一次 Checkpoint
        if n % 1000 == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "ckpt"))

    # 保存最终 Checkpoint
    checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "final_ckpt"))

    return deprocess_image(generated_image.numpy())