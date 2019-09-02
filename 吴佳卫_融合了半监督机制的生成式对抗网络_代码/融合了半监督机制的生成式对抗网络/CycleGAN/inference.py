"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb /
                       --input input_sample.jpg /
                       --output output_sample.jpg /
                       --image_size 48
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'pretrained/A2B.pb', 'model path (.pb)')
tf.flags.DEFINE_integer('image_size', '48', 'image size, default: 48')

load_file_path = "../Data/Data_test/classified_data/A"
save_file_path = "../Data/Data_test/cycle_GAN_result/A"
all_files = []


def inference(g_input, g_output):
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(g_input, 'rb') as f:
      image_data = f.read()
      input_image = tf.image.decode_jpeg(image_data, channels=3)
      input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
      input_image = utils.convert2float(input_image)
      input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='g_output')

  with tf.Session(graph=graph) as sess:
    generated = output_image.eval()
    with open(g_output, 'wb') as f:
      f.write(generated)


for root, dirs, files in os.walk(load_file_path):
    for file in files:
        if "jpg" in file:
            all_files.append(file)

final_img_num = int(len(all_files))
for n in range(0, final_img_num):
    g_input = load_file_path + "/" + all_files[n]
    g_output = save_file_path + "/" + all_files[n]
    inference(g_input, g_output)
