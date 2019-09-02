from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
import os


class DataReader:
    def __init__(self, sess, train_dir, image_dimension, output_vector_size, num_channels=3):
        self.sess = sess
        # Get filenames
        train_filepaths, train_labels = self.load_data(train_dir, output_vector_size)

        # Convert to tensors
        train_filepaths_tensor = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)

        # Create queues
        train_input_queue = tf.train.slice_input_producer(
                                    [train_filepaths_tensor, train_labels],
                                    shuffle=True)

        # Process string tensor to image
        train_file_content = tf.read_file(train_input_queue[0])
        self.train_image = tf.image.decode_jpeg(train_file_content, channels=num_channels)
        self.train_label = train_input_queue[1]

        # Define tensor shape
        self.train_image.set_shape([image_dimension, image_dimension, num_channels])

    def get_train_batch(self, coord, batch_size=100, num_threads=8):
        train_batch = tf.train.batch(
            [self.train_image, self.train_label],
            batch_size=batch_size,
            num_threads=num_threads
        )
        tf.train.start_queue_runners(sess=self.sess, coord=coord)
        batch = self.sess.run(train_batch)
        return batch[0], batch[1]

    @staticmethod
    def load_data(directory, vector_size):
        # Get all subdirectories of data_dir. Each represents a label.
        directories = [d for d in os.listdir(directory)
                       if os.path.isdir(os.path.join(directory, d))]
        # Loop through the label directories and collect the data in
        # two lists, labels and filenames.
        labels = []
        filenames = []
        for d in directories:
            label_dir = os.path.join(directory, d)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir)]
            for f in file_names:
                filenames.append(f)
                labels_one_hot = []
                for i in range(vector_size):
                    labels_one_hot.append(1 if int(d) == i else 0)
                labels.append(labels_one_hot)
        return filenames, labels
