"""Builds the CNN network.
structure from the tensorflow example :
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

Implements the tensorflow inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.
"""

import tensorflow as tf
import math
from tools import *


def placeholders_openai() -> tf.Tensor:
    """
    :return: the placeholders
    """
    image_placeholder = tf.placeholder(tf.float32, shape=(1, 84, 84, 4), name='input_image')
    return image_placeholder


def placeholders_training(batch_size: int) -> (tf.Tensor, tf.Tensor):
    """
    :param batch_size: number of images in the batch
    :return: the placeholders
    """
    image_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 4), name='input_image')
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size, name='Input_label')

    return image_placeholder, labels_placeholder


def inference(image_placeholder: tf.placeholder, num_action: int, batch_size: int) -> tf.Tensor:
    """
    Build the CNN model up to where it may be used for inference.
    The model is built according to the original UCTtoCNN paper

    :param image_placeholder:
    :param num_action: number of possible actions
    :return: tensor with the computed logits
    """

    h = convolution_layer_tanh([8, 8, 4, 16], [1, 4, 4, 1], image_placeholder)
    h = convolution_layer_tanh([4, 4, 16, 32], [1, 2, 2, 1], h)
    h = tf.reshape(h, [batch_size, 11 * 11 * 32])
    h = fully_connected_relu(11 * 11 * 32, 256, h)

    with tf.name_scope('final_layer'):
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=[256, num_action], stddev=0.1), name='Weights')
        biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=[num_action], dtype=tf.float32), name='biases')

        logits = tf.matmul(h, weights) + biases
        # softact = tf.nn.softmax(logits, name='softact')

    return logits


def classification_loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)  # from the website TODO: tester l'autre
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # from the website TODO: tester l'autre
    # optimizer = tf.train.RMSPropOptimizer(learning_rate * (tf.pow(0.9, (global_step / 1000))),
    #                                       decay=0.9)  # from VIN implementation
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1e-6, centered=True)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy
