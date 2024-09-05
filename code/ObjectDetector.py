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

X = data.iloc[1:, 1:-1].values
y = data.iloc[1:, -1]


class ObjectDetector:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.clf = Classifier(self.X, self.y)
        self.clf.train_model()
        
        
    #sliding windows which slide through every pixel in the images
    def sliding_window(self, img, window_sizes, stride):
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
    def pyramid(self, img, scale=0.8, min_size=(30, 30)):
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
    def visualize_bbox(self,img_filename, img, bboxes, label_encoder):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for box in bboxes:
            xmin, ymin, xmax, ymax, predict_id, conf_score = box
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            classname = label_encoder.inverse_transform([predict_id])[0]
            label = f"{classname} {conf_score}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            cv2.rectangle(img, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)
            
            cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('data\\detect\\result\\' + img_filename, img_bgr)
        print(img_filename)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        
    
    #compute IoU function for NMS calculation
    def compute_iou(self, bbox, bboxes, bbox_area, bboxes_area):
        xxmin = np.maximum(bbox[0], bboxes[:, 0])
        yymin = np.maximum(bbox[1], bboxes[:, 1])
        xxmax = np.maximum(bbox[2], bboxes[:, 2])
        yymax = np.maximum(bbox[3], bboxes[:, 3])
        
        w = np.maximum(0, xxmax - xxmin + 1)
        h = np.maximum(0, yymax - yymin + 1)
        
        intersection = w * h
        iou = intersection / (bbox_area + bboxes_area - intersection)
        
        return iou
    
    
    #compute Non-Maximum Suppression
    def nms(self, bboxes, iou_threshold):
        if not bboxes:
            return []
        
        scores = np.array([bbox[5] for bbox in bboxes])
        sorted_indices = np.argsort(scores)[::-1]
        
        xmin = np.array([bbox[0] for bbox in bboxes])
        ymin = np.array([bbox[1] for bbox in bboxes])
        xmax = np.array([bbox[2] for bbox in bboxes])
        ymax = np.array([bbox[3] for bbox in bboxes])
        
        areas = (xmax - xmin + 1) * (ymax - ymin + 1)
        
        keep = []
        
        while sorted_indices.size > 0:
            i = sorted_indices[0]
            keep.append(i)
            
            iou = self.compute_iou(
                [xmin[i], ymin[i], xmax[i], ymax[i]],
                np.array(
                    [
                        xmin[sorted_indices[1:]],
                        ymin[sorted_indices[1:]],
                        xmax[sorted_indices[1:]],
                        ymax[sorted_indices[1:]]
                    ]
                ).T,
                areas[i],
                areas[sorted_indices[1:]]
            )
            idx_to_keep = np.where(iou <= iou_threshold)[0]
            sorted_indices = sorted_indices[idx_to_keep + 1]
            
        return [bboxes[i] for i in keep]
            
    
        
    #detect the object in the image file    
    def detect(self, img_dir):
        img_filename_lst = os.listdir(img_dir)[:20]
        conf_threshold = 0.9
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
                    stride=stride
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
                        
            bboxes = self.nms(bboxes, 0.5)
            self.visualize_bbox(img_filename, img, bboxes, self.clf.label_encoder)
            
            
detectObj = ObjectDetector(X, y)
detectObj.detect('data\\detect\\images_to_detect')