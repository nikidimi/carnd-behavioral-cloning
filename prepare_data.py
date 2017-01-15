import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from preprocess import image_preprocess

class DataPreperator:
    X_train = []
    y_train = []
    directory = '/'
    
    def set_prepend_directory(self, directory):
        self.directory = directory + '/'
    
    def _prepend_path(self, filename):
        return self.directory + '/' + filename
    
    def _append_to_list(self, path, angle, flip = False, brightness_adjust = 1.0):
        self.X_train.append((self._prepend_path(path), flip, brightness_adjust))
        if flip:
            angle = angle * -1
        self.y_train.append(angle)
    
    def _append_brightness_adjusts(self, path, angle, flip, brightness_adjusts = [1.0]):
        for brightness in brightness_adjusts:
            self._append_to_list(path, angle, flip, brightness)
    
    def _append_adjusted_images(self, row):
        angle = float(row[3])
        ANGLE_ADJUST = 0.25
        
        #self._append_to_list(row[0], angle)
        
        if abs(float(row[3])) > 0.01:
            self._append_brightness_adjusts(row[0], angle, False)
            self._append_brightness_adjusts(row[0], angle, True)
        else:
            self._append_to_list(row[0], angle)
            
        self._append_brightness_adjusts(row[1], angle + ANGLE_ADJUST, False)
        self._append_brightness_adjusts(row[2], angle - ANGLE_ADJUST, False)
        self._append_brightness_adjusts(row[1], angle + ANGLE_ADJUST, True)
        self._append_brightness_adjusts(row[2], angle - ANGLE_ADJUST, True)
        

        
    def read_csv(self, directory_name, absolute_path = True):
        if not absolute_path:
            self.set_prepend_directory(directory_name)
        
        with open(directory_name + '/driving_log.csv', newline='') as csvfile:
            logreader = csv.reader(csvfile, delimiter=',', quotechar='|', skipinitialspace=True)
            for row in logreader:
                self._append_adjusted_images(row)

def save_image_data(paths, labels, batch_size=32):
    X_train = np.empty([len(paths), 16, 32, 1])
    y_train = np.empty([len(paths)])
    for index in range(0, len(paths)):

        
        image = plt.imread(paths[index][0])            
        arr = image_preprocess(image, paths[index][2])
        if (paths[index][1]):
            arr = cv2.flip(arr, 1)
                
        X_train[index] = arr.reshape((16, 32, 1))
        y_train[index] = labels[index]


    np.save("x.data", X_train)
    np.save("y.data", y_train)

if __name__ == '__main__':
    data_preparator = DataPreperator()
    data_preparator.read_csv(directory_name = 'udacity', absolute_path=False)
    
    
    save_image_data(data_preparator.X_train, data_preparator.y_train)

