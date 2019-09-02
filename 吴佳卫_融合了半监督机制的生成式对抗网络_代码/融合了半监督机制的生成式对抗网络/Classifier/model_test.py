import tensorflow as tf
import layers as ly
import test_data_loader as tdl
import output_data as od

""" Input image dimension (square) """
INPUT_IMAGE_DIMENSION = 48

""" Input image channel (greyscale) """
INPUT_IMAGE_CHANNELS = 3


def conv_net_model_test(save_dir, test_dir, output_dir0, output_dir1):
    """
    The feed forward convolutional neural network model

    Hyper parameters include learning rate, number of convolutional layers and
    fully connected layers. (Currently TBD)

    """
    # Reset graphs
    tf.reset_default_graph()

    # Create placeholders
    x = tf.placeholder(dtype=tf.float32,
                       shape=[None, INPUT_IMAGE_DIMENSION, INPUT_IMAGE_DIMENSION, INPUT_IMAGE_CHANNELS],
                       name="x")

    weight1 = tf.Variable(tf.truncated_normal([4, 4, 3, 16], stddev=0.1), dtype=tf.float32, name="W1")
    bias1 = tf.Variable(tf.constant(0.1, shape=[16]), dtype=tf.float32, name="B1")
    weight2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev=0.1), dtype=tf.float32, name="W2")
    bias2 = tf.Variable(tf.constant(0.1, shape=[32]), dtype=tf.float32, name="B2")
    weight3 = tf.Variable(tf.truncated_normal([4608, 2], stddev=0.1), dtype=tf.float32, name="W3")
    bias3 = tf.Variable(tf.constant(0.1, shape=[2]), dtype=tf.float32, name="B3")

    # First convolutional layer
    conv1 = ly.conv_layer(x, weight1, bias1, False)

    # First pooling
    pool1 = ly.pool_layer(conv1)

    # Second convolutional layer
    conv2 = ly.conv_layer(pool1, weight2, bias2, True)

    # Second pooling
    pool2 = ly.pool_layer(conv2)

    # Flatten input
    flattened = tf.reshape(pool2, shape=[-1, 12 * 12 * 32])

    # Create fully connected layer
    logits = ly.fully_connected_layer(flattened, weight3, bias3)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_dir)
        # Run model
        test_images = tdl.load_test_data(test_dir)

        coord = tf.train.Coordinator()

        # Test the model
        l = sess.run(tf.argmax(logits, 1), feed_dict={x: test_images})
        od.output(output_dir0, output_dir1, test_images, l)

    coord.request_stop()
