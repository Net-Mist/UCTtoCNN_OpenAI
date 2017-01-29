"""
python 3.6.0
the global organization of the code is highly inspire from :
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
"""

from cnn_atari import *
from dataset_management import *
import time
import os
import argparse
import os.path
import tensorflow as tf

# program flags. For more details see the end of this file
FLAGS = None

BLOCK_SIZE = 8  # Should be a divisor of 160
NUMBER_BLOCK = int(160 / BLOCK_SIZE)
K_REC = 40


def fill_feed_dict(category, image_lists, image_pl, labels_pl):
    images, labels = get_random_cached_images(image_lists, FLAGS.batch_size, category)

    feed_dict = {
        image_pl: images,
        labels_pl: labels
    }
    return feed_dict


def do_eval(sess, eval_correct, image_pl, labels_placeholder, category, image_lists, writer, summary, step):
    """
    Runs one evaluation against the full epoch of data.
    :param sess: The session in which the model has been trained.
    :param eval_correct: The Tensor that returns the number of correct predictions.
    :param image_pl: placeholder for the image
    :param labels_placeholder: The labels placeholder.
    :param category:
    :param image_lists:
    """
    feed_dict = fill_feed_dict(category, image_lists, image_pl, labels_placeholder)
    accuracy = sess.run(eval_correct, feed_dict=feed_dict)
    summary_str = sess.run(summary, feed_dict=feed_dict)
    writer.add_summary(summary_str, step)
    writer.flush()

    print(F"Step: {step}, category : {category}, accuracy: {accuracy}")

    # # Compute the number of data to test
    # num_examples = 0
    # for label_name in image_lists:
    #     num_examples += len(image_lists[label_name][category])
    # if num_examples > FLAGS.eval_step_number:
    #     num_examples = FLAGS.eval_step_number
    # steps_per_epoch = num_examples // FLAGS.batch_size
    # num_examples = steps_per_epoch * FLAGS.batch_size
    #
    # if num_examples == 0:
    #     num_examples = 1

    # true_count = 0  # Counts the number of correct predictions.
    # for step in range(steps_per_epoch):
    #     feed_dict = fill_feed_dict(category, image_lists, image_pl, labels_placeholder)
    #     true_count += sess.run(eval_correct, feed_dict=feed_dict)
    # precision = float(true_count) / num_examples
    # print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
    #       (num_examples, true_count, precision))


def main():
    # Look at the folder structure, and create lists of all the images.
    image_lists = load_data(FLAGS.testing_percentage, FLAGS.validation_percentage)

    print('number of labels : ', len(image_lists))

    with tf.Graph().as_default():
        print('Create network placeholders')
        image_placeholder, labels_placeholder = placeholders_training(FLAGS.batch_size)
        print('Create the inference part')
        logits = inference(image_placeholder, 3, FLAGS.batch_size)

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

        # Instantiate a SummaryWriter to output summaries and the Graph.
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        print('INIT ALL !!!!!')
        sess.run(init)

        class_count = len(image_lists)
        if class_count == 0:
            print('No valid folders of images found at ' + FLAGS.image_dir)
        if class_count == 1:
            print('Only one valid folder of images found at ' + FLAGS.image_dir +
                  ' - multiple classes are needed for classification.')

        # Start the training loop.
        for step in range(FLAGS.how_many_training_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict('training', image_lists, image_placeholder, labels_placeholder)

            # print(feed_dict)
            # print("plip")
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            # print("plop")

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                do_eval(sess, eval_correct, image_placeholder, labels_placeholder, 'training', image_lists,
                        train_writer, summary, step)
                # # Print status to stdout.
                # print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # # Update the events file.
                # summary_str = sess.run(summary, feed_dict=feed_dict)
                # train_writer.add_summary(summary_str, step)
                # train_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % FLAGS.eval_step_interval == 0 or (step + 1) == FLAGS.how_many_training_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                # print('Training Data Eval:')

                # Evaluate against the validation set.
                # print('Validation Data Eval:')
                # do_eval(sess, eval_correct, images_placeholder, labels_placeholder, 'validation', image_lists)
                # Evaluate against the test set.
                # print('Test Data Eval:')
                do_eval(sess, eval_correct, image_placeholder, labels_placeholder, 'testing', image_lists, test_writer,
                        summary, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--how_many_training_steps', type=int, default=200000,
                        help='How many training steps to run before ending.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='How large a learning rate to use when training.')
    parser.add_argument('--testing_percentage', type=int, default=20,
                        help='What percentage of images to use as a test set.')
    parser.add_argument('--validation_percentage', type=int, default=0,
                        help='What percentage of images to use as a validation set.')
    parser.add_argument('--eval_step_interval', type=int, default=100,
                        help='How often to evaluate the training results.')
    parser.add_argument('--batch_size', type=int, default=50, help='How many images to train on at a time.')
    parser.add_argument('--log_dir', type=str, default='/tmp/TensorFlow-VIN',
                        help='Path to folders to log training.')
    FLAGS = parser.parse_args()
    main()
    # tf.app.run() TODO analyser ce que ça fait
