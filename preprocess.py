import keras.preprocessing.image as kim
import cv2

def image_preprocess(image, brightness=1.0):
    """
    Resizes one image, applies the needed brightness_adjust and converts to HSV colour space
    Normalizes the data to [-1 to 1]

    Parameters
    ----------
    image : numpy array
        The image
    brightness: float
        brightness adjusment factor
    Returns
    -------
    image : numpy array
        The processed image
    """
    arr = image_brightness_adjust(image, brightness)
    arr = cv2.resize(arr, (32, 16))
    arr = arr / 127.5 - 1
    return arr

def image_brightness_adjust(image, brightness):
    """
    Convert an image to HSV colour space, apply brightness adjustment

    Parameters
    ----------
    image : numpy array
        The image
    brightness: float
        brightness adjusment factor
    Returns
    -------
    image : numpy array
        The S channel of the image
    """
    arr = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    arr[:, :, 2] = arr[:, :, 2] * brightness
    return arr[:, :, 1]
