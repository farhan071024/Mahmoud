#!/usr/bin/env python
import input, math, flags as f
import tensorflow as tf
def model_fn(sample_input = None, dropout_rate = 0.5, is_training = True, summaries = None):
      CONV_FILTER_DEPTH = input.SAMPLE_DEPTH
      CONV_FILTER_HEIGHT = 3
      CONV_FILTER_WIDTH = 3
      with tf.name_scope('reshape'):
            input_layer = tf.reshape(sample_input, [-1, input.SAMPLE_DEPTH, input.SAMPLE_HEIGHT, input.SAMPLE_WIDTH, 1])
      
      #First convolutional layer - maps samples to 32 feature map
      with tf.name_scope('conv1'):
            weights = tf.Variable(tf.truncated_normal([CONV_FILTER_DEPTH, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT, 1, 32], stddev=0.1))
            conv1 = tf.nn.conv3d(
                  input=input_layer,
                  filter=weights,
                  strides=[1, 1, 1, 1, 1],
                  padding="SAME")
            if f.FLAGS.batch_normalization:
                  conv1 = tf.contrib.layers.batch_norm(
                        conv1,
                        data_format='NHWC',
                        center=True,
                        scale=True,
                        updates_collections=None,
                        is_training=is_training,
                        scope='conv1-batch_norm')
                  conv1_act = tf.nn.relu(conv1)
            else:
                  bias = tf.Variable(tf.constant(0.1, shape=[32]))
                  conv1_act = tf.nn.relu(conv1 + bias)
                  summaries['train'].append(tf.summary.histogram("bias", bias))
            summaries['train'].append(tf.summary.histogram("weights", weights))
            summaries['train'].append(tf.summary.histogram("conv", conv1))
            summaries['train'].append(tf.summary.histogram("activation", conv1_act))

      
      #Pool layer - downsamples by a factor of 2
      with tf.name_scope('pool1'):
            ksize = [1, 1, 2, 2, 1]
            strides = [1, 1, 2, 2, 1]
            pool1 = tf.nn.max_pool3d(conv1_act, ksize=ksize, strides=strides, padding="SAME")

      #Second convolutional layer - maps 32 feature maps to 64
      with tf.name_scope('conv2'):
            weights = tf.Variable(tf.truncated_normal([CONV_FILTER_DEPTH, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT, 32, 64], stddev=0.1))
            conv2 = tf.nn.conv3d(
                  input=pool1,
                  filter=weights,
                  strides=[1, 1, 1, 1, 1],
                  padding="SAME")
            if f.FLAGS.batch_normalization:
                  conv2 = tf.contrib.layers.batch_norm(
                        conv2,
                        data_format='NHWC',
                        center=True,
                        scale=True,
                        updates_collections=None,
                        is_training=is_training,
                        scope='conv2-batch_norm')
                  conv2_act = tf.nn.relu(conv2)
            else:
                  bias = tf.Variable(tf.constant(0.1, shape=[64]))
                  conv2_act = tf.nn.relu(conv2 + bias)
                  summaries['train'].append(tf.summary.histogram("bias", bias))

            summaries['train'].append(tf.summary.histogram("weights", weights))
            summaries['train'].append(tf.summary.histogram("conv", conv2))
            summaries['train'].append(tf.summary.histogram("activation", conv2_act))

      #Pool layer - downsize by factor of 2
      with tf.name_scope('pool2'):
            ksize = [1, 1, 2, 2, 1]
            strides = [1, 1, 2, 2, 1]
            pool2 = tf.nn.max_pool3d(conv2_act, ksize=ksize, strides=strides, padding="SAME")
            
      #Fully connected layer 1
      with tf.name_scope('fc1'):
            #new_depth = int(math.ceil(math.ceil(input.SAMPLE_DEPTH / 2.0) / 2.0))
            new_depth = input.SAMPLE_DEPTH
            new_height = int(math.ceil(math.ceil(input.SAMPLE_HEIGHT / 2.0) / 2.0))
            new_width = int(math.ceil(math.ceil(input.SAMPLE_WIDTH / 2.0) / 2.0))
            weights = tf.Variable(tf.truncated_normal([new_depth * new_height * new_width * 64, 1024], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[1024]))
            
            pool2_flat = tf.reshape(pool2, [-1, new_depth * new_height * new_width * 64])
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + bias)
            summaries['train'].append(tf.summary.histogram("weights", weights))
            summaries['train'].append(tf.summary.histogram("bias", bias))
            summaries['train'].append(tf.summary.histogram("fc1", fc1))

            if f.FLAGS.dropout:
                  with tf.name_scope('dropout1'):
                        fc1 = tf.nn.dropout(fc1, dropout_rate)

      with tf.name_scope('fc2'):
            weights = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[512]))
            fc2 = tf.nn.relu(tf.matmul(fc1, weights) + bias)
            summaries['train'].append(tf.summary.histogram("weights", weights))
            summaries['train'].append(tf.summary.histogram("bias", bias))
            summaries['train'].append(tf.summary.histogram("fc2", fc2))
            if f.FLAGS.dropout:
                  with tf.name_scope('dropout2'):
                        fc2 = tf.nn.dropout(fc2, dropout_rate)            
                  
      with tf.name_scope('fc_pred'):
            weights = tf.Variable(tf.truncated_normal([512, 2], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[2]))
            fc_pred = tf.matmul(fc2, weights) + bias
            summaries['train'].append(tf.summary.histogram("weights", weights))
            summaries['train'].append(tf.summary.histogram("bias", bias))
            summaries['train'].append(tf.summary.histogram("fc_pred", fc_pred))

      return fc_pred
