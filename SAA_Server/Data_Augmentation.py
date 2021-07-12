# Importing necessary modules

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import os
import glob
import cv2
import tqdm


def start_data_augmentation():
    

    # Initialising the ImageDataGenerator class.
    # We will pass in the augmentation parameters in the constructor.
    
    datagen = ImageDataGenerator(rotation_range = 5,
            zoom_range = 0.2,
            shear_range = 0.2,
            horizontal_flip = True,
            brightness_range = (0.5, 1.5))
    
    # Load all images from the directory
    path_list = glob.glob("final_dataset/*.png")
        
    # Loading image by image from the list
    iterator = 0
    for path in tqdm(path_list):
        img = load_img(path)
        save_img('augmented_dataset/image_{}.png'.format(iterator), img)
        # Converting the input image to an array
        x = img_to_array(img)
        # Reshaping the input image
        x = x.reshape((1, ) + x.shape) 
       
        # Generating and saving 5 augmented samples 
        # using the above defined parameters. 
        i = 0
        for batch in datagen.flow(x, batch_size = 1, save_to_dir ='augmented_dataset', save_prefix ='image', save_format ='png'):
            i += 1
            if i > 10:
                break
        iterator += 1
    print('Finished Augmentation!')

def rename_images():
    
    # Rename all the images
    for count, filename in enumerate(os.listdir("augmented_dataset")):
        dst ="image" + str(count) + ".png"
        src ='augmented_dataset/'+ filename
        dst ='augmented_dataset/'+ dst
        
        # rename() function will
        # rename all the files
        os.rename(src, dst)

def main():
    # Create directory to ssave the augmented images if not exists
    if not os.path.exists('augmented_dataset'):
        os.makedirs('augmented_dataset')
    
    # The augmentation function may take a while to finish execution
    
    # start_data_augmentation()
    
    rename_images()

main()
