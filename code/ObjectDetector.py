import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import time

from preprocess_data import preprocess_img
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
    
    
    #visualize_bbox on the image
    def visualize_bbox(img, bboxes, label_encoder):
        img = cv2.cvtColor(img, cv2.BGR2RGB)
        
        for box in bboxes:
            xmin, ymin, xmax, ymax, predict_id, conf_score = box
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            classname = label_encoder.inverse_transform([predict_id])[0]
            label = f"{classname} {conf_score}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            cv2.rectangle(img, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)
            
            cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        
    def detect(self, img_dir):
        img_filename_lst = os.listdir(img_dir)[:20]
        conf_threshold = 0.8
        stride = 12
        window_sizes = [
            (32, 32),
            (64, 64),
            (128, 128)
        ]
        
        #iterate through each images in the directory
        for img_filename in img_filename_lst:
            start_time = time.time()
            img_filepath = os.path.join(img_dir, img_filename)
            bboxes = []
            img = cv2.imread(img_filepath)
            pyramid_imgs = self.pyramid(img)
            
            #iterate each images in pyramid and run sliding windows
            for pyramid_img_info  in pyramid_imgs:
                pyramid_img, scale_factor = pyramid_img_info
                window_lst = self.sliding_window(
                    pyramid_img,
                    window_sizes=window_sizes,
                    stride=stride,
                    scale_factor=scale_factor
                )
                
                #iterate each sliding windows and 
                for window in window_lst:
                    xmin, ymin, xmax, ymax = window
                    object_img = pyramid_img[ymin:ymax, xmin:xmax]
                    preprocessed_img = preprocess_img(object_img)
                    normalized_img = self.clf.scaler.transform([preprocessed_img])[0]
                    decision = self.clf.model.predict_proba([normalized_img])[0]
                    
                    if np.all(decision < conf_threshold):
                        continue
                    else:
                        predict_id = np.argmax(decision)
                        conf_score = decision[predict_id]
                        xmin = int(xmin / scale_factor)
                        ymin = int(ymin / scale_factor)
                        xmax = int(xmax / scale_factor)
                        ymax = int(ymax / scale_factor)
                        bboxes.append(
                            [xmin, ymin, xmax, ymax, predict_id, conf_score]
                        )