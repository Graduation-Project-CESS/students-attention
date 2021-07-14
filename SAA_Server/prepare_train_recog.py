import os
import numpy as np


def rename_images(directory):
    
    # Rename all the images
    for count, filename in enumerate(os.listdir(os.path.join(root_directory,directory))):
        dst = directory + ' ' + str(count) + ".png"
        src = os.path.join(root_directory,directory, filename)
        dst = root_directory +'/' + dst
        
        # rename() function will
        # rename all the files
        os.rename(src, dst)


root_directory = 'augmented_cropped_images'
files = []
directories = []


for _, dirnames, filenames in os.walk(root_directory):
    # ^ this idiom means "we won't be using this value"
    directories.append(dirnames)
        
directories = np.squeeze([x for x in directories if x != []])

for directory in directories:
    print('In directory: {}'.format(directory))
    rename_images(directory)


print('Finished Preparing the training dataset')