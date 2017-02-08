"""
python 3.6.0
the global organization of the code is highly inspire from :
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
"""

from cnn_atari import *
from dataset_management_cluster import *
import time
import os
import argparse
import os.path
import tensorflow as tf
import random

# program flags. For more details see the end of this file
FLAGS = None


def fill_feed_dict(category, image_lists, image_pl, labels_pl, keep_prob_placeholder, keep_prob):
    images, labels, init_index = get_random_cached_images(image_lists, FLAGS.batch_size, category)

    feed_dict = {
        image_pl: images,
        labels_pl: labels,
        keep_prob_placeholder: keep_prob
    }
    return feed_dict, init_index


def do_eval(sess, eval_correct, feed_dict, category, writer, summary, step):
    """
    Runs one evaluation against the full epoch of data.
    :param sess: The session in which the model has been trained.
    :param eval_correct: The Tensor that returns the number of correct predictions.
    :param image_pl: placeholder for the image
    :param labels_placeholder: The labels placeholder.
    :param category:
    :param image_lists:
    """

    accuracy = sess.run(eval_correct, feed_dict=feed_dict)
    summary_str = sess.run(summary, feed_dict=feed_dict)
    writer.add_summary(summary_str, step)
    writer.flush()

    print(F"Step: {step}, category : {category}, accuracy: {accuracy}")


def main():
    with tf.Graph().as_default():

        # Create the neural network
        print('Create network placeholders')
        image_placeholder, labels_placeholder, keep_prob_placeholder = placeholders_training(FLAGS.batch_size)
        print('Create the inference part')
        logits = inference(image_placeholder, keep_prob_placeholder, 3, FLAGS.batch_size)
        print('Create the training part')
        loss = classification_loss(logits, labels_placeholder)
        train_op = training(loss, FLAGS.learning_rate)
        print('Create the evaluation part')
        eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate SummaryWriters to output summaries and the Graph.
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)

        # Run the Op to initialize the variables.
        print('Init the neural network')
        sess.run(init)

        global_step = 0
        for epoch_i in range(FLAGS.how_many_epochs):
            print("epoch n'" + str(epoch_i + 1))
            files_to_load = list(range(1, FLAGS.total_number_file + 1))

            # if there is still files to load
            while files_to_load:
                # select a certain amount of files in the list files_to_load. The random seed is fixed because if
                # we work with multiple epoch, we need to have the same testing and training set.
                files_loaded = []
                for _ in range(min(FLAGS.number_file_per_iteration, len(files_to_load))):
                    np.random.seed(files_to_load)
                    index = np.random.randint(len(files_to_load))
                    files_loaded.append(files_to_load[index])
                    files_to_load = files_to_load[:index] + files_to_load[index + 1:]

                # Load the data and split them randomly between training, testing and validation set
                image_lists = None
                image_lists = load_data(files_loaded, FLAGS.files_dir, FLAGS.testing_percentage,
                                        FLAGS.validation_percentage)

                # Start the training loop.
                how_many_training_steps = int(len(image_lists['training']['data']) / FLAGS.batch_size)

                for step in range(how_many_training_steps):
                    start_time = time.time()
                    feed_dict, init_index = fill_feed_dict('training', image_lists, image_placeholder,
                                                           labels_placeholder,
                                                           keep_prob_placeholder, 1.0)

                    if init_index:
                        print("reinitialise cached images for training set")

                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                    duration = time.time() - start_time
                    # print("duration : ", duration)

                    # Write the summaries and save a checkpoint fairly often.
                    if (step + 1) % FLAGS.eval_step_interval == 0 or (step + 1) == how_many_training_steps:
                        # Save
                        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=global_step)

                        # Eval training
                        do_eval(sess, eval_correct, feed_dict, 'training', train_writer, summary, global_step)

                        # Eval testing
                        feed_dict, init_index = fill_feed_dict('testing', image_lists, image_placeholder,
                                                               labels_placeholder,
                                                               keep_prob_placeholder, 1.0)
                        do_eval(sess, eval_correct, feed_dict, 'testing', test_writer, summary, global_step)
                        if init_index:
                            print("reinitialise cached images for testing set")
                    global_step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--how_many_epochs', type=int, default=3,
                        help='How many epochs to run before ending.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='How large a learning rate to use when training.')
    parser.add_argument('--testing_percentage', type=int, default=1,
                        help='What percentage of images to use as a test set.')
    parser.add_argument('--validation_percentage', type=int, default=0,
                        help='What percentage of images to use as a validation set.')
    parser.add_argument('--eval_step_interval', type=int, default=100,
                        help='How often to evaluate the training results.')
    parser.add_argument('--batch_size', type=int, default=50, help='How many images to train on at a time.')
    parser.add_argument('--log_dir', type=str, default='/tmp/TensorFlow-CNN',
                        help='Path to folders to log training.')
    # For the cluster
    parser.add_argument('--files_dir', type=str, default='/hpctmp2/e0046667/CNN',
                        help='Path to the folder which contain the npz files.')
    parser.add_argument('--total_number_file', type=int, default=50,
                        help='The total number of file with will be use for the training.')
    parser.add_argument('--number_file_per_iteration', type=int, default=50,
                        help='The maximum number of files that can be load in the memory.')

    FLAGS = parser.parse_args()
    random.seed(time.time())
    main()
