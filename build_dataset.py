# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Modified version of build_image_data.py from tensorflow/models/inception/inception/data
Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in tif files located in the
following directory structure.
  ...
  train/1_1.tif
  train/1_1_mask.tif
  train/2_1.tif
  train/2_1_mask.tif
  ...

where the the tif file whose filename has 'mask' is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  output_directory/train-00000-of-0124
  output_directory/train-00001-of-0124
  ...
  output_directory/train-00127-of-0124

and

  output_directory/test-00000-of-00128
  output_directory/test-00001-of-00128
  ...
  output_directory/test-00127-of-00128

where we have selected 128 and 32 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

    'image/height': integer, image height in pixels
    'image/width': integer, image width in pixels
    'image/colorspace': string, specifying the colorspace, always 'RGB'
    'image/channels': integer, specifying the number of channels, always 3
    'image/format': string, specifying the format, always'JPEG'
    'image/filename': string containing the basename of the image file
            e.g. '1_1.tif' or '2_1.tif'
    'image/encoded': string containing JPEG encoded image in RGB colorspace
    'image/label/height': integer, label image height in pixels
    'image/label/width': integer, label image width in pixels
    'image/label/colorspace': string, specifying the colorspace, always 'RGB'
    'image/label/channels': integer, specifying the number of channels, always 3
    'image/label/format': string, specifying the format, always'JPEG'
    'image/label/filename': string containing the basename of the image file
            e.g. '1_1_mask.tif' or '2_1_mask.tif'
    'image/label/encoded': string containing JPEG encoded image in RGB colorspace

Note: label height, width, colorspace, channels, format can be excluded from that
proto but it only takes about an additional 1 MB more space, so is included in the
proto
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import glob
import cv2
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train_directory', '/home/luna/PycharmProjects/NerveSegmentation/raw_data/train',
                           'Training data directory.')
tf.app.flags.DEFINE_float('train_split_percent', 0.8,
                          'Percentage of data used to train on from the total train data supplied, '
                          'rest used to test.')
tf.app.flags.DEFINE_integer('train_shards', 128,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 32,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_string('output_directory', '/home/luna/PycharmProjects/NerveSegmentation/processed_dataset',
                           'Output data directory')

FLAGS = tf.app.flags.FLAGS


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts TIFF to JPEG data.
        self._tiff_data = tf.placeholder(dtype=tf.uint8)
        self._tiff_to_jpeg = tf.image.encode_jpeg(self._tiff_data, quality=100)

        self._rgb_to_grayscale_data = tf.image.rgb_to_grayscale(self._tiff_data)
        self._rgb_to_grayscale_jpeg = tf.image.encode_jpeg(self._rgb_to_grayscale_data, quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        self._decode_grayscale_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_grayscale_jpeg = tf.image.decode_jpeg(self._decode_grayscale_jpeg_data, channels=1)

    def tiff_to_jpeg(self, image_data, is_grayscale=False):
        if is_grayscale:
            return self._sess.run(self._rgb_to_grayscale_jpeg,
                                  feed_dict={self._tiff_data: image_data})
        else:
            return self._sess.run(self._tiff_to_jpeg,
                                  feed_dict={self._tiff_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def decode_grayscale_jpeg(self, image_data):
        image = self._sess.run(self._decode_grayscale_jpeg,
                               feed_dict={self._decode_grayscale_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image


def _is_tiff(filename):
    """Determine if a file contains a TIFF format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a TIFF.
    """
    return '.tif' in filename


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width, label, label_image_buffer,
                        label_image_height, label_image_width):
    """
    Build an example proto for an example
    :param filename: string, path to an training image file, e.g., '/path/to/example.TIFF'
    :param image_buffer: string, JPEG encoding of RGB image
    :param height: integer, image height in pixels
    :param width: integer, image width in pixels
    :param label: string, path to a label image file, e.g., '/path/to/example.TIFF'
    :param label_image_buffer: string, JPEG encoding of RGB image
    :param label_image_height: integer, image height in pixels
    :param label_image_width: integer, image width in pixels
    :return: Example proto
    """
    colorspace = 'RGB'
    colorspace_grayscale = 'GRAYSCALE'
    channels = 3
    grayscale_channels = 1
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer),
        'image/label/height': _int64_feature(label_image_height),
        'image/label/width': _int64_feature(label_image_width),
        'image/label/colorspace': _bytes_feature(colorspace_grayscale),
        'image/label/channels': _int64_feature(grayscale_channels),
        'image/label/format': _bytes_feature(image_format),
        'image/label/filename': _bytes_feature(os.path.basename(label)),
        'image/label/encoded': _bytes_feature(label_image_buffer)}))

    return example


def _process_image(filename, coder, is_label=False):
    """
    Process a single image file.
    :param filename: string, path to an image file e.g., '/path/to/example.JPG'.
    :param coder: instance of ImageCoder to provide TensorFlow image coding utils.
    :return:    image_buffer: string, JPEG encoding of RGB image.
                height: integer, image height in pixels.
                width: integer, image width in pixels.
    """
    # Read the image file.
    image_data = cv2.imread(filename)

    if is_label:
        r1, g1, b1 = 255, 255, 255
        r2, g2, b2 = 1, 1, 1
        red, green, blue = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        image_data[:, :, :3][mask] = [r2, g2, b2]
        image_data = image_data[10:410, 20:580]
        temp_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY).flatten()
        count_of_zeros_and_ones = np.bincount(temp_img.flatten())
        num_label_zero.append(count_of_zeros_and_ones[0])
        num_label_one.append(count_of_zeros_and_ones[1])
        image_data = coder.tiff_to_jpeg(image_data, is_grayscale=True)
    else:
        image_data = image_data[10:410, 20:580]
        image_data = coder.tiff_to_jpeg(image_data)

    # Decode the RGB JPEG.
    if is_label:
        image = coder.decode_grayscale_jpeg(image_data)
        # Check that image converted to RGB
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 1
    else:
        image = coder.decode_jpeg(image_data)
        # Check that image converted to RGB
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, split_images_list, split_labels_list,
                               num_shards):
    """
    Processes and saves list of images as TFRecord in 1 thread.
    :param coder: instance of ImageCoder to provide TensorFlow image coding utils.
    :param thread_index: integer, unique batch to run index is within [0, len(ranges)).
    :param ranges: list of pairs of integers specifying ranges of each batches to analyze in parallel.
    :param name: string, unique identifier specifying the data set
    :param split_images_list: list of strings; each string is a path to an image file
    :param split_labels_list: list of strings; each string is a path to an image file
    :param num_shards: integer number of shards for this data set.
    :return: None
    """
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = split_images_list[i]
            label = split_labels_list[i]

            image_buffer, height, width = _process_image(filename, coder)
            label_image_buffer, label_image_height, label_image_width = _process_image(label, coder, is_label=True)
            example = _convert_to_example(filename, image_buffer, height, width,
                                          label, label_image_buffer, label_image_height, label_image_width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, split_images_list, split_labels_list, num_shards):
    """
    Process and save list of images as TFRecord of Example protos.
    :param name: string, unique identifier specifying the data set
    :param split_images_list: list of strings; each string is a path to an image file
    :param split_labels_list: list of strings; each string is a path to an image file
    :param num_shards: integer number of shards for this data set.
    :return: None
    """
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(split_images_list), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, split_images_list,
                split_labels_list, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d %s images in data set.' %
          (datetime.now(), len(split_images_list), name))
    sys.stdout.flush()


def _process_dataset(train_split_images_list, train_split_labels_list, test_split_images_list,
                     test_split_labels_list):
    """
    Process a complete data set and save it as a TFRecord.
    :param train_split_images_list: list of training images
    :param train_split_labels_list: list of labels for training images
    :param test_split_images_list: list of test images
    :param test_split_labels_list: list of labels for test images
    :return: None
    """
    _process_image_files('train', train_split_images_list, train_split_labels_list, FLAGS.train_shards)
    _process_image_files('test', test_split_images_list, test_split_labels_list, FLAGS.test_shards)


def _create_train_test_split(training_images_list, label_images_list, train_split_percent):
    """
    Divides the training images folder into training images and test images, with train_split
    percent being the percent of images in training set and the rest in test set. Note that,
    the Nerve Segmentation Challenge gives a set of images in the 'test' folder. That folder
    is not touched at all during the training process. It is referred to as the validation
    data throughout the documentation of this code. Test images in our case is a fraction of
    the images in the 'train' folder
    :param training_images_list: list of images from the training folder
    :param label_images_list: list of label images from the training folder
    :param train_split_percent: percentage of images used for training
    :return:
    """
    assert len(training_images_list) == len(label_images_list), ('Number of training images and label '
                                                                 'images must be same. Please make sure '
                                                                 'equal number of training images and '
                                                                 'label images')

    split_index = int(len(training_images_list) * train_split_percent)
    train_split_images_list = training_images_list[0:split_index]
    train_split_labels_list = label_images_list[0:split_index]
    test_split_images_list = training_images_list[split_index:len(training_images_list)]
    test_split_labels_list = label_images_list[split_index:len(training_images_list)]
    print('Finished splitting data into %s training images and %s '
          'test images' % (len(train_split_images_list), len(test_split_images_list)))
    return train_split_images_list, train_split_labels_list, test_split_images_list, test_split_labels_list


def _create_image_list(image_directory):
    """
    Create lists of training images and their corresponding labels
    :param image_directory: directory where the training images and labels are
    :return:
        training_images: list containing file path of training images
        label_images: list containing file path of labelled images
    """
    print('Creating list of training images and a list of corresponding labels from %s' % image_directory)
    training_images = []
    label_images = []
    for n in glob.glob('%s/*[0-9].tif' % image_directory):
        training_images.append(n)
        label_images.append(n.strip('.tif') + '_mask.tif')

    print('Shuffling the list of images and labels')
    shuffled_index = range(len(training_images))
    random.seed(12345)
    random.shuffle(shuffled_index)
    training_images = [training_images[i] for i in shuffled_index]
    label_images = [label_images[i] for i in shuffled_index]
    print('Finished creating list of %s images and %s labels' % (len(training_images), len(label_images)))
    return training_images, label_images


def main(unused_argv):
    total_label_count = 400 * 560
    num_label_zero = 0
    num_label_one = 0

    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.test_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.test_shards')

    print('Saving results to %s' % FLAGS.output_directory)

    training_images_list, label_images_list = _create_image_list(FLAGS.train_directory)
    train_split_images_list, train_split_labels_list, test_split_images_list, test_split_labels_list \
        = _create_train_test_split(training_images_list, label_images_list, FLAGS.train_split_percent)
    _process_dataset(train_split_images_list, train_split_labels_list, test_split_images_list,
                     test_split_labels_list)

    print ("Probabilities of label 0: %s. Please put it in nerve_data.py" % (np.sum(num_label_zero).astype(np.float32) / total_label_count))
    print("Probabilities of label 1: %s. Please put it in nerve_data.py" % (np.sum(num_label_one).astype(np.float32) / total_label_count))


if __name__ == '__main__':
    tf.app.run()
