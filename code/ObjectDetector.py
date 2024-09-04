import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from Classifier import Classifier


#read preprocessed data
data = pd.read_csv('data\\preprocessed_imgs.csv', header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1]


class ObjectDetector:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.clf = Classifier(self.X, self.y)
        
        
    #sliding windows which slide through every pixel in the images
    def sliding_window(img, window_sizes, stride):
        img_height, img_width = img.shape[:2]
        windows = []
        for window_size in window_sizes:
            window_width, window_height = window_size
            for ymin in range(0, img_height - window_height + 1, stride):
                for xmin in range(0, img_width - window_width + 1, stride):
                    xmax = xmin + window_width
                    ymax = ymin + window_height
                    
                    windows.append([xmin, ymin, xmax, ymax])
                    
        return windows
    
    
    #Pyramid Image technique to find small objects
    def pyramid(img, scale=0.8, min_size=(30, 30)):
        acc_scale = 1.0
        pyramid_images = [(img, acc_scale)]
        
        while True:
            acc_scale = acc_scale * scale
            h = int(img.shape[0] * acc_scale)
            w = int(img.shape[1] * acc_scale)
            if w < min_size[0] or h < min_size[1]:
                break
            
            img = cv2.resize(img, (w, h))
            pyramid_images.append((img, acc_scale))
            
        return pyramid_images