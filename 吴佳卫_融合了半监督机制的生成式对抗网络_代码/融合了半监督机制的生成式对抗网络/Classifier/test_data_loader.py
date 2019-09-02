import tensorflow as tf
import os
import numpy as np


def load_test_data(data_path):

    imgs = os.listdir(data_path)
    imgNum = len(imgs)
    test_images = np.empty((imgNum, 48, 48, 3), dtype="float32")

    with tf.Session() as sess:
        for i in range(imgNum):
            img_content = tf.read_file(data_path + "/" + imgs[i])
            arr = sess.run(tf.image.decode_jpeg(img_content, channels=3))
            test_images[i, :, :, :] = arr
    return test_images




