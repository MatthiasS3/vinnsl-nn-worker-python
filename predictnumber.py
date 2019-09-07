# Import Modules
import numpy as np
from keras.models import *
from keras import backend as K

import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import center_of_mass


def load(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

def nnPredict(image):
    if image is not None:
        model = load_model("dnn-mnist.h5")
        pred = model.predict(image.reshape(1, 784))[0]
        pred = np.argmax(pred, axis=0)
        #Clear Session
        K.clear_session()
        return pred

def getImage():
    image = load("image.png").astype(np.float32)  # image ist jetzt ein array
    #image = normalize(image)
    image = correct(image)
    image = center(image)
    image = resize(image)
    return image

def resize(image):
    image = cv2.resize(image, (28, 28))
    return image

def normalize(image):
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return image

def correct(image):
    image[:, 0] = 0.0
    image[:, -1] = 0.0
    image[0, :] = 0.0
    image[-1, :] = 0.0
    return image

def center(image):
    cy, cx = center_of_mass(image)
    rows, cols = image.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def predictNumber():
    image = getImage()
    pred = nnPredict(image=image)
    return int(pred)

