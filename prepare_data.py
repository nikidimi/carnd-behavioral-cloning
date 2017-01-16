import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from preprocess import image_preprocess

class DataPreparator:
    """
    This class reads a CSV file and prepares a list of the paths of the images and specifies what augmentations to apply later on
    It does NOT read images or apply transformations. It only generates a list with the required transformations
    This allows the usage of generator when reading images
    
    Attributes
    ----------
    X_train : list of tuples
        A list of tuples - (image path, should we flip this images, brightness adjust factor)
    y_train : list
        The angles for the each image are stored here. Same indexing as X_train
    """
    X_train = []
    y_train = []
    directory = '/'
    
    def set_prepend_directory(self, directory):
        """
        Changes what is the directory name we should prepend to the path from the CSV file
        Used when parsing CSV files with relative paths.
        
        Parameters
        ----------
        directory : string
            The path to append in front
        """
        self.directory = directory + '/'
    
    def _prepend_path(self, filename):
        return self.directory + '/' + filename
    
    
    def _append_to_list(self, path, angle, flip = False, brightness_adjust = 1.0):
        """
        Adds one image to the X/y_train list
        
        Parameters
        ----------
        path : string
            The path to the image
        angle : float
            Steering angle
        flip : boolean
            Should we flip the image when reading the list
        brightness_adjusts: float
            Brightness adjust factor
        """
        self.X_train.append((self._prepend_path(path), flip, brightness_adjust))
        if flip: #Flipped images angle needs correction
            angle = angle * -1
        self.y_train.append(angle)


    def _append_brightness_adjusts(self, path, angle, flip, brightness_adjusts = [1.0]):
        """
        Specifies what the needed brightness adjusts for a specific image are and appends them to X/y_train
        
        Parameters
        ----------
        path : string
            The path to the image
        angle : float
            Steering angle
        flip : boolean
            Should we flip the image when reading the list
        brightness_adjusts: list
            List of floats specifying what brightness adjusts to add
        """
        for brightness in brightness_adjusts:
            self._append_to_list(path, angle, flip, brightness)
    
    def _append_adjusted_images(self, row):
        """
        Specifies what the needed transformations for a specific row are and appends them to X/y_train
        
        Parameters
        ----------
        row : dict
            A row from the CSV file
        """
        angle = float(row[3])
        ANGLE_ADJUST = 0.25 # How much to adjust angle for left/right camera
        
        # Add the image from the central camera
        if abs(float(row[3])) > 0.01: #Do not flip images with angle close to 0. This helps balance the data
            self._append_brightness_adjusts(row[0], angle, False)
            self._append_brightness_adjusts(row[0], angle, True)
        else:
            self._append_to_list(row[0], angle)
        
        # Add images from left/right camera
        self._append_brightness_adjusts(row[1], angle + ANGLE_ADJUST, False)
        self._append_brightness_adjusts(row[2], angle - ANGLE_ADJUST, False)
        # Add images from left/right camera, flipped
        self._append_brightness_adjusts(row[1], angle + ANGLE_ADJUST, True)
        self._append_brightness_adjusts(row[2], angle - ANGLE_ADJUST, True)
        

        
    def read_csv(self, directory_name, absolute_path = True):
        """
        Reads a CSV file in the Udacity driving log format
        Goes over each row and adds the information to X_train and y_train
        
        Parameters
        ----------
        directory_name : string
            The directory where drivinig_log.csv and the images are stored
        absolute_path: boolean
            Should we prepend the directory_name to the paths in the CSV file or are they absolute paths
        """
        if not absolute_path:
            self.set_prepend_directory(directory_name)
        
        with open(directory_name + '/driving_log.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile, delimiter=',', quotechar='|', skipinitialspace=True)
            for row in logreader:
                self._append_adjusted_images(row)

def save_image_data(paths, labels):
    """
    This functions reads lists in the DataPreparator format, opens an image, 
    applies transformations and saves them in numpy binary format

    Parameters
    ----------
    paths : string
        The path to append in front
        
    """
    # Output data format
    X_train = np.empty([len(paths), 16, 32, 1])
    y_train = np.empty([len(paths)])
    
    # Go over all images
    for index in range(0, len(paths)):
        image = plt.imread(paths[index][0])        
        
        # Apply the needed transformations    
        arr = image_preprocess(image, paths[index][2])
        if (paths[index][1]):
            arr = cv2.flip(arr, 1)
        
        # Store the data for output, reshaping it from 16X32 to 16X32X1
        X_train[index] = arr.reshape((16, 32, 1))
        y_train[index] = labels[index]

    # Save the data in numpy binary format. 
    # This speeds up training because we need to do image parsing only once, not before every training
    np.save("x.data", X_train)
    np.save("y.data", y_train)

if __name__ == '__main__':
    # Open the CSV file and parse it
    data_preparator = DataPreparator()
    data_preparator.read_csv(directory_name = 'udacity', absolute_path=False)
    
    # Save the image data in numpy binary files
    save_image_data(data_preparator.X_train, data_preparator.y_train)

