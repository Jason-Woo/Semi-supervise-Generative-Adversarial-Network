import tensorflow as tf
import layers as ly
import datareader as dr

""" Input image dimension (square) """
INPUT_IMAGE_DIMENSION = 48

""" Input image channel (greyscale) """
INPUT_IMAGE_CHANNELS = 3

""" Training step size """
STEP_SIZE = 500

""" Input batch size"""
BATCH_SIZE = 80

""" Output one hot vector size """
OUTPUT_VECTOR_SIZE = 2


def conv_net_model_train(learning_rate, train_dir, save_dir):
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
    y = tf.placeholder(dtype=tf.float32,
                       shape=[None, OUTPUT_VECTOR_SIZE],
                       name="y")
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

    # Create loss function
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # Create optimizer
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Compute accuracy
    with tf.name_scope("accuracy"):
        # argmax gets the highest value in a given dimension (in this case, dimension 1)
        # equal checks if the label is equal to the computed logits
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        # tf.reduce_mean computes the mean across the vector
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run model
        sess.run(tf.global_variables_initializer())
        data_reader = dr.DataReader(sess,
                                    train_dir,
                                    INPUT_IMAGE_DIMENSION,
                                    OUTPUT_VECTOR_SIZE,
                                    INPUT_IMAGE_CHANNELS)

        coord = tf.train.Coordinator()

        # Train the model
        for i in range(STEP_SIZE):
            images, labels = data_reader.get_train_batch(coord, BATCH_SIZE)

            if i % 10 == 0:
                a = sess.run(accuracy, feed_dict={x: images, y: labels})

                print("step", i, "of ", STEP_SIZE)
                print("Acc: ", a)

            # Run the training step
            sess.run(train_step, feed_dict={x: images, y: labels})

        saver.save(sess, save_dir)

    coord.request_stop()
