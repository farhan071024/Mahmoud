import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('window_size', 1, "In case of 3d CNN, provide a window size. window_size = 1 is the same as 2d CNN.")
tf.app.flags.DEFINE_integer('height', 120, "Height of a single sample.")
tf.app.flags.DEFINE_integer('width', 45, "Width of a single sample.")
tf.app.flags.DEFINE_integer('batch_size', 64, "Number of sample in a single batch.")
tf.app.flags.DEFINE_integer('num_epochs', 20, "Number of training epochs.")
tf.app.flags.DEFINE_string('data_dir', '~/tfrecords/*', "TFRecords directory.")
tf.app.flags.DEFINE_float('learning_rate', 1e-5, "Optimizer learning rate.")
tf.app.flags.DEFINE_string('log_dir', 'log/e1', "Location of saving logs")
tf.app.flags.DEFINE_string('cnn_model', 'lenet5', "CNN model to use")
tf.app.flags.DEFINE_boolean('dropout', False, "Enable dropout with 0.5")
tf.app.flags.DEFINE_boolean('batch_normalization', False, "Enable batch normalization")
