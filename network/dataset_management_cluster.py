"""
highly inspire from the bottleneck cached technique in :
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
"""
import numpy as np


def load_data(files_to_load: [], files_dir: str, testing_percentage: float, validation_percentage: float) -> {}:
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    :param files_to_load: the array of the number of the npz files to load
    :param files_dir: Directory where the npz are stored
    :param testing_percentage: Integer percentage of the images to reserve for tests.
    :param validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """

    # Load all the files and compute the total amount of data
    # (it's dirty, think of something else ? A text file will these information ? A metadata npz with a list with the
    # number of data TODO)
    total_number_data = 0
    files = []
    for file_number in files_to_load:
        file_name = files_dir + '/data' + str(file_number) + '.npz'
        print('load', file_name)
        file = np.load(file_name)
        total_number_data += file['nb_frames'][0]
        files.append(file)

    # Create the numpy array to store the data
    images = np.zeros((total_number_data, 84, 84, 4), dtype='B')
    labels = np.zeros(total_number_data, dtype='B')

    # fill the numpy array
    current_frame = 0
    for file in files:
        print(current_frame, "out of", total_number_data)
        images[current_frame:current_frame + file['images'].shape[0], :, :, :] = file['images']
        labels[current_frame:current_frame + file['images'].shape[0]] = file['image_index_to_action_index']
        current_frame += file['images'].shape[0]

    # shuffle the array with a specific seed so that the training and testing set are always the same
    # with the same files
    np.random.seed(files_to_load)
    perm = np.random.permutation(total_number_data)
    images = images[perm]
    labels = labels[perm]

    # Prepare the result
    image_lists = {
        'training': {
            'data': [],
            'label': [],
            'current_index': 0
        },
        'testing': {
            'data': [],
            'label': [],
            'current_index': 0
        },
        'validation': {
            'data': [],
            'label': [],
            'current_index': 0
        }
    }

    # Fill the result
    for i in range(len(images)):
        random_percentage = np.random.randint(100)
        if random_percentage < validation_percentage:
            image_lists['validation']['data'].append(images[i])
            image_lists['validation']['label'].append(labels[i])
        elif random_percentage < (testing_percentage + validation_percentage):
            image_lists['testing']['data'].append(images[i])
            image_lists['testing']['label'].append(labels[i])
        else:
            image_lists['training']['data'].append(images[i])
            image_lists['training']['label'].append(labels[i])

    print(str(len(image_lists['training']['data'])) + ' training images, ' + str(
        len(image_lists['testing']['data'])) + ' testing images ' + str(
        len(image_lists['validation']['data'])) + ' validation images.')
    return image_lists


def get_random_cached_images(image_lists: {}, how_many: int, category: str) -> ([], [], bool):
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
    init_index = False
    for _ in range(how_many):
        images.append(image_lists[category]['data'][image_lists[category]['current_index']])
        labels.append(image_lists[category]['label'][image_lists[category]['current_index']])
        image_lists[category]['current_index'] += 1

        if image_lists[category]['current_index'] >= len(image_lists[category]['data']):
            image_lists[category]['current_index'] = 0
            init_index = True

    return images, labels, init_index
