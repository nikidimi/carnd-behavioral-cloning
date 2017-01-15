import keras.preprocessing.image as kim
import cv2

def image_preprocess(image, brightness=1.0):
    arr = image_brightness_adjust(image, brightness)
    arr = cv2.resize(arr, (32, 16))
    arr = arr / 127.5 - 1
    return arr

def image_brightness_adjust(image, brightness):
    arr = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    arr[:, :, 2] = arr[:, :, 2] * brightness
    return arr[:, :, 1]
