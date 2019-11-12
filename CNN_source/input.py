from __future__ import print_function
import os, sys, glob, json, numpy as np, random, time, gc
import tensorflow as tf
import flags as f

SAMPLE_WIDTH = f.FLAGS.width
SAMPLE_HEIGHT = f.FLAGS.height
SAMPLE_DEPTH = f.FLAGS.window_size

def _parse_fn(batch_record):
    features = {
        "sample": tf.FixedLenFeature([SAMPLE_DEPTH * SAMPLE_HEIGHT * SAMPLE_WIDTH], tf.float32),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_example(batch_record, features)
    data = parsed_features["sample"]
    data = tf.reshape(data, [f.FLAGS.batch_size, SAMPLE_DEPTH, SAMPLE_HEIGHT, SAMPLE_WIDTH])
    labels = parsed_features["label"]
    labels = tf.one_hot(labels, 2)
    return data, labels

def _get_dataset(files, batch_size, shuffle_data):
    dataset = tf.data.TFRecordDataset(files)
    if shuffle_data:
        dataset = dataset.shuffle(buffer_size=20000)
    #batch first here for performance issue "bug"
    dataset = dataset.batch(batch_size, drop_remainder=True)
    #num_parallel_calls should be less than num of CPU cores
    dataset = dataset.map(_parse_fn, num_parallel_calls=8)
    return dataset

def input(shuffle_files=False):
    files = glob.glob(f.FLAGS.data_dir + "/*.tfrecord")
    if len(files) == 0:
        exit("TFRecords directory is empty")
        
    if shuffle_files:
        random.shuffle(files)

    #files = files[:20]
    #Dividing data
    train_files = files[:int(0.6*len(files))]
    validate_files = files[int(0.6*len(files)):int(0.80*len(files))]
    test_files = files[int(0.80*len(files)):]
    
    print ("Number of training files: ", len(train_files))
    print ("Number of validation files: ", len(validate_files))
    print ("Number of testing files: ", len(test_files))
    
    train_dataset = _get_dataset(files=train_files, batch_size=f.FLAGS.batch_size, shuffle_data=True)
    validate_dataset = _get_dataset(files=validate_files, batch_size=f.FLAGS.batch_size, shuffle_data=False)
    test_dataset = _get_dataset(files=test_files, batch_size=f.FLAGS.batch_size, shuffle_data=False)

    train_dataset = train_dataset.prefetch(1)
    validate_dataset = validate_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)
    
    return train_dataset, validate_dataset, test_dataset
