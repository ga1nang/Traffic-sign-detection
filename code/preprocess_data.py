import time
import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import csv

from skimage.transform import resize
from skimage import feature


#preprocessing image function hog
def preprocess_img(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    
    resized_img = resize(
        img,
        output_shape=(32, 32),
        anti_aliasing=True
    )
    
    hog_feature = feature.hog(
        resized_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2",
        feature_vector=True
    )
    
    return hog_feature

'''read data from the .xml file and .png file'''
annotations_dir = 'data\\annotations'
img_dir = 'data\\images'

img_lst = []
label_lst = []

#iterate through all file
for xml_file in os.listdir(annotations_dir):
    #read image
    xml_filepath = os.path.join(annotations_dir, xml_file)
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    img_filename = root.find('filename').text
    img_filepath = os.path.join(img_dir, img_filename)
    img = cv2.imread(img_filepath)
    
    #crop traffic sign from image based on data from .xml file
    for obj in root.findall('object'):
        classname = obj.find('name').text
        if classname == 'trafficlight':
            continue
        
        label_lst.append(classname)
        
        #crop image
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        obj_image = img[ymin:ymax, xmin:xmax]
        img_lst.append(obj_image)
        
        
#preprocess the cropped image with HOG
img_features_lst = []
for img in img_lst:
    hog_feature = preprocess_img(img)
    img_features_lst.append(hog_feature)
img_features = np.array(img_features_lst)

#write data to a seperate file   
df = pd.DataFrame(img_features)
df['label'] = label_lst
df.to_csv('data\\preprocessed_imgs.csv', header=None)