"""
Reference:
Martin Gorner's TPU introduction notebook: https://t.ly/IT9G
"""

import tensorflow as tf
import numpy as np
import re

GCS_PATTERN = "gs://flowers-public/tfrecords-jpeg-224x224/*.tfrec"
AUTO = tf.data.AUTOTUNE

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def create_steps(batch_size, validation_split=0.15):
    validation_split = 0.15
    filenames = tf.io.gfile.glob(GCS_PATTERN)
    split = len(filenames) - int(len(filenames) * validation_split)
    train_filenames = filenames[:split]
    valid_filenames = filenames[split:]
    train_steps = count_data_items(train_filenames) // batch_size
    return (train_filenames, valid_filenames, train_steps)

def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),
        # shape [] means scalar
        "one_hot_class": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    class_label = tf.cast(example['class'], tf.int32)
    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])
    one_hot_class = tf.reshape(one_hot_class, [5])
    return image, one_hot_class

def force_image_sizes(dataset, image_size=(96, 96, 3)):
    # explicit size will be needed for TPU
    reshape_images = lambda image, label: (tf.reshape(image, image_size), label)
    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)
    return dataset

def load_dataset(filenames):
    # read from TFRecords. For optimal performance, use "interleave(tf.data.TFRecordDataset, ...)"
    # to read from multiple TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.
    opt = tf.data.Options()
    opt.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(opt)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset)
    return dataset

def data_augment(image, one_hot_class):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0, 2)
    return image, one_hot_class

def get_training_dataset(batch_size, filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_validation_dataset(batch_size, filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset