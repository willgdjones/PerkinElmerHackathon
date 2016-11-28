import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import os
import pdb
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected, convolution2d, max_pool2d, flatten
import time
from helpers import *

image_list = os.listdir('png/size1080/images/')
image_IDs = [x.split('.')[0] for x in image_list]