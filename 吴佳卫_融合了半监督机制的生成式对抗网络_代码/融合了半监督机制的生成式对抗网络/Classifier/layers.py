import tensorflow as tf


def conv_layer(batch_input, weight, bias, activate=False):

    conv = tf.nn.conv2d(batch_input, weight, strides=[1, 1, 1, 1], padding="SAME")
    if activate is True:
        act = tf.nn.relu(conv + bias)
    else:
        act = conv + bias
    return act


def pool_layer(batch_input):

    return tf.nn.max_pool(batch_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fully_connected_layer(flattened_input, weight, bias):

    act = tf.matmul(flattened_input, weight) + bias
    return act
