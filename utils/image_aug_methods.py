# Reference https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

import tensorflow as tf
import numpy as np
import cv2


##############################################################
#                           Scale                            #
##############################################################


def central_scale_images(X_imgs, IMAGE_SIZE, scales):
    """
    :param X_imgs: (m, height, width, channel)
    :param IMAGE_SIZE: (height, width)
    :param scales: list of scale factor in range (0, 1), e.g. [0.75, 0.9]
    :return: numpy array of scaled images. Same size of X_imgs
    """

    height, width = IMAGE_SIZE
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([height, width], dtype=np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, height, width, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_imgs)

    X_scale_data = np.array(X_scale_data, dtype=np.float32)
    return X_scale_data


##############################################################
#                           Flip                            #
##############################################################


def flip_images(X_imgs, IMAGE_SIZE):
    """
    :param X_imgs: (m, height, width, channel)
    :param IMAGE_SIZE: (height, width)
    :return: numpy array of scaled images. Same size of X_imgs
    """
    height, width = IMAGE_SIZE
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(height, width, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    # tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2], feed_dict={X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.asarray(X_flip, dtype=np.float32)
    return X_flip


##############################################################
#                     Lighting Condition                     #
##############################################################


def add_gaussian_noise(X_imgs):
    """
    :param X_imgs: (m, height, width, channel)
    :return: numpy array of scaled images. Same size of X_imgs
    """
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape

    # Gaussian distribution parameters
    # mean = 0
    # var = 0.1
    # sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs