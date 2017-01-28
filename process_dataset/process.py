import numpy as np
from PIL import Image
import glob
import time

output_dir = '../outputs_12h_1000_100'
data = []

simulation_dir = glob.glob(output_dir + '/*')
simulation_dir.sort()
nb_simulations = len(simulation_dir)
# print(simulation_dir)  # ['../outputs_12h_1000_100/output1', '../outputs_12h_1000_100/output2',

for simulation_path in simulation_dir:
    action_dirs = glob.glob(simulation_path + '/dataset/*')
    action_dirs.sort()
    nb_actions = len(action_dirs)

    # Compute the number of frame
    nb_frames = 0
    for action_dir in action_dirs:
        nb_frames += len(glob.glob(action_dir + '/*'))
    print(nb_frames, "frames")

    images = np.zeros((nb_frames, 84, 84, 4), dtype=int)
    image_index_to_action_index = np.zeros(nb_frames, dtype=int)

    i = 0
    for action_dir in action_dirs:
        image_paths = glob.glob(action_dir + '/*')
        image_paths.sort()
        for image_path in image_paths:
            # Prepare the image
            image = Image.open(image_path).convert('L')
            image = image.crop((0, 34, 160, 194))
            image = image.resize((84, 84))
            image = np.array(image)

            # Save the image in 4 buffer
            frame_number = int(image_path.split('/')[-1][:-4])
            images[frame_number - 1, :, :, 0] = image
            if frame_number < nb_frames:
                images[frame_number, :, :, 1] = image
            if frame_number + 1 < nb_frames:
                images[frame_number + 1, :, :, 2] = image
            if frame_number + 2 < nb_frames:
                images[frame_number + 2, :, :, 3] = image

            # and save the action
            image_index_to_action_index[frame_number - 1] = i # int(action_dir.split('/')[-1])
        i += 1

    np.savez('../data' + simulation_path.split('/')[-1][-1], images=images,
             image_index_to_action_index=image_index_to_action_index)
