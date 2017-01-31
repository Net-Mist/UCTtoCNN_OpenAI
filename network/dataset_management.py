"""
highly inspire from the bottleneck cached technique in :
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
"""
import random
import numpy as np
import glob

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def load_data(testing_percentage: float, validation_percentage: float) -> {}:
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    :param testing_percentage: Integer percentage of the images to reserve for tests.
    :param validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    # Load the npz files
    images = []
    labels = []

    dataset_files = glob.glob('/home/mist/Projects/Dissertation/data_from_cluster/*.npz')
    dataset_files.sort()

    # We need to know the total amount of data (it's dirty..)
    print("Computing the total amount of data. We can have this information in the processing")
    # for dataset_file in dataset_files:
    #     print(dataset_file)
    #     file = np.load(dataset_file)
    total = 81671
    images = np.zeros((total, 84, 84, 4), dtype=np.float16)
    labels = np.zeros(total, dtype='b')

    actuel = 0
    for dataset_file in dataset_files:
        print(dataset_file)
        file = np.load(dataset_file)

        images[actuel:actuel+file['images'].shape[0], :, :, :] = file['images']
        labels[actuel:actuel+file['images'].shape[0]] = file['image_index_to_action_index']

        actuel += file['images'].shape[0]

        # print(images.shape)
        # print(labels.shape)

    # Subtract the mean and scale the input to lie in [-1,1] TODO : substract the real mean...
    # print("Convert to float64")
    # images = np.array(images, dtype=np.float16)
    # print("Scale")
    # images = images / (255 / 2)
    # print("Center")
    # images = images - 1

    # Prepare the result
    result = {}
    for i in range(3):
        result[i] = {
            'training': [],
            'testing': [],
            'validation': [],
        }

    for i in range(len(images)):
        random_percentage = random.randrange(100)
        if random_percentage < validation_percentage:
            result[labels[i]]['validation'].append(images[i])
        elif random_percentage < (testing_percentage + validation_percentage):
            result[labels[i]]['testing'].append(images[i])
        else:
            result[labels[i]]['training'].append(images[i])

    training_nb = 0
    testing_nb = 0
    validation_nb = 0
    for i in result:
        training_nb += len(result[i]['training'])
        testing_nb += len(result[i]['testing'])
        validation_nb += len(result[i]['validation'])
    print(str(training_nb) + ' training images, ' + str(testing_nb) + ' testing images ' + str(
        validation_nb) + ' validation images.')
    return result


def get_random_cached_images(image_lists: {}, how_many: int, category: str) -> ([], [], []):
    """
    Args:
        image_lists: Dictionary of training images for each label.
        how_many: The number of bottleneck values to return.
        category: Name string of which set to pull from - training, testing, or validation.
    Returns:
        List of bottleneck arrays and their corresponding ground truths.
    """

    images = []
    labels = []
    for _ in range(how_many):
        # randomly chose the class among all
        label_index = random.randrange(3)

        # randomly chose an image among the chosen class
        image_index = random.randrange(len(image_lists[label_index][category]))
        image = image_lists[label_index][category][image_index]

        # here we don't need to to this because we're using tf.nn.sparse_softmax_cross_entropy_with_logits
        # ground_truth = np.zeros(class_count, dtype=np.float32)
        # ground_truth[label_index] = 1.0
        images.append(image)
        labels.append(label_index)

    return images, labels
