import gym
from gym import wrappers
from cnn_atari import *
import tensorflow as tf
import numpy as np
from PIL import Image
import time

ATARI_SCREEN_WIDTH = 210
ATARI_SCREEN_HEIGHT = 160
NUM_ACTIONS = 3


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


index_to_action = [0, 3, 4]


def main():
    previous_images = np.zeros((84, 84, 4), dtype='b')
    print("Init OpenAI part")
    # env = gym.make('PongNoFrameskip-v3')
    env = gym.make('Pong-v3')
    env = wrappers.Monitor(env, '/tmp/pong_experiment', force=True)
    print(env.action_space)

    env.frameskip = 3

    observation = env.reset()
    # print(observation.shape)  # (210, 160, 3)
    with tf.Graph().as_default():
        print('Create neural network')
        images_placeholder, keep_prob_placeholder = placeholders_openai()
        logits = inference(images_placeholder, keep_prob_placeholder, NUM_ACTIONS, 1)
        sess = tf.Session()
        saver = tf.train.Saver()
        # saver.restore(sess, "../TensorFlow-VIN/model.ckpt-102899")
        saver.restore(sess, "/tmp/TensorFlow-CNN/model.ckpt-189402")

        print("Launch game")
        for i in range(1000000):
            env.render()
            # process the current game image
            game_image = Image.fromarray(observation).convert('L').crop((0, 34, 160, 194)).resize((84, 84))
            game_image = np.array(game_image, dtype='B')

            previous_images[:, :, 3] = previous_images[:, :, 2]
            previous_images[:, :, 2] = previous_images[:, :, 1]
            previous_images[:, :, 1] = previous_images[:, :, 0]
            previous_images[:, :, 0] = game_image

            l = sess.run([logits], feed_dict={images_placeholder: [previous_images], keep_prob_placeholder: 1})
            # observation, reward, done, info = env.step(labels[np.argmax(l)])
            # print(l[0][0])
            l = softmax(l[0][0])
            action = index_to_action[np.argmax(l)]
            observation, reward, done, info = env.step(action)
            # print(l, action)
            # time.sleep(0.1)
            # print(l)
            if done:
                print("Episode finished")
                break


if __name__ == '__main__':
    main()
