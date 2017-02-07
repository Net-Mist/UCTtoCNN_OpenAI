import tensorflow as tf


def convolution_layer_tanh(kernel_shape, strides, input):
    """
    :param kernel_shape: for instance [8, 8, 4, 8]
    :param strides: for instance [1, 4, 4, 1]
    """
    with tf.name_scope('Convolution'):
        kernel = tf.Variable(initial_value=tf.truncated_normal(shape=kernel_shape, stddev=0.1), name='Kernel')
        biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=[kernel_shape[3]], dtype=tf.float32),
                             name='biases')
        conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
        h = tf.tanh(tf.nn.bias_add(conv, biases))
    return h


def convolution_layer_relu(kernel_shape, strides, input):
    """
    :param kernel_shape: for instance [8, 8, 4, 8]
    :param strides: for instance [1, 4, 4, 1]
    """
    with tf.name_scope('Convolution'):
        kernel = tf.Variable(initial_value=tf.truncated_normal(shape=kernel_shape, stddev=0.1), name='Kernel')
        biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=[kernel_shape[3]], dtype=tf.float32),
                             name='biases')
        conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
        h = tf.nn.relu(tf.nn.bias_add(conv, biases))
    return h

def fully_connected_relu(input_size, output_size, input):
    with tf.name_scope('fully_connected'):
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=[input_size, output_size], stddev=0.1),
                              name='Weights')
        biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=[output_size], dtype=tf.float32), name='biases')
        h = tf.nn.relu(tf.matmul(input, weights) + biases)
    return h
