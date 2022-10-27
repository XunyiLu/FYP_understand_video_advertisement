import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from keras.preprocessing import image  
from tensorflow.keras.utils import load_img, img_to_array

import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


for i in tqdm(range(1)):
    count = 0
    #videoFile = train['video_name'][i]
    cap = cv2.VideoCapture("/Users/prochetasen/Documents/Research/VideoClassification/01_Preliminaries.mp4")   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename ="/Users/prochetasen/Documents/Research/VideoClassification/pics/_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()


