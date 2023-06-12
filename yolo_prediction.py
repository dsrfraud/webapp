#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import torchvision
import cv2
import numpy as np
import numpy as np

from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import cv2
import re
import numpy as np



from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray

from scipy.stats import mode


# In[14]:


import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd=r'C:\Users\ACER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# In[15]:


model_path = r'/home/ubuntu/webapp/models/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom',path=model_path)


# In[16]:


# image_path = r"D:\Pan\pandata\final_data\pancard_1.jpeg"


# image = rgb2gray(imread(image_path))
# edges = canny(image)
# # Classic straight-line Hough transform
# tested_angles = np.deg2rad(np.arange(0.1, 180.0))
# h, theta, d = hough_line(edges, theta=tested_angles)



def rotate_image(image, angle, resize=False):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    if resize:
        # calculate the new image size based on the rotation angle
        height, width = rotated.shape[:2]
        angle_rad = angle * np.pi / 180
        new_width = abs(width * np.cos(angle_rad)) + abs(height * np.sin(angle_rad))
        new_height = abs(width * np.sin(angle_rad)) + abs(height * np.cos(angle_rad))
        new_size = (int(new_width), int(new_height))
        # resize the rotated image to the new size
        rotated = cv2.resize(rotated, new_size)
    return rotated

def rotate90(image):
    timage = cv2.transpose(image)
    flipped = cv2.flip(timage, 1)
    return flipped


def skew_angle_hough_transform(image):
    
    # convert to edges
    edges = canny(image)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    print(f'Skew Angle {skew_angle}')
    return skew_angle


# skew = skew_angle_hough_transform(image)


# img = rotate_image(image, skew[0]+180)

def is_valid_pan(pan):
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'
    return bool(re.match(pattern, pan))

def is_valid_name(name):
    pattern = "^[A-Za-z]+(([',. -][A-Za-z ])?[A-Za-z]*)*$"
    return bool(re.match(pattern, name))

def is_valid_dob(dob):
    pattern = r'^(0[1-9]|[1-2][0-9]|3[0-1])/(0[1-9]|1[0-2])/([0-9]{4})$'
    return bool(re.match(pattern, dob))

def get_ocr(real_img, predictions):
    result_dict ={}
    
    for i, pred in enumerate(predictions.iterrows()):
        label  = pred[1][6]

        index, data = pred
        bbox = data[["xmin", "ymin", "xmax", "ymax"]].tolist()
        # Cast bbox coordinates to integers
        bbox = [int(x) for x in bbox]
        cropped_image = real_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = Image.fromarray(cropped_image)

        img = np.array(img) 
        
        # Convert RGB to BGR 
        img = img[:, :, ::-1].copy()
        h,w,c = img.shape
        img = cv2.resize(img, (w,h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(img,lang='eng', config='--psm 7')
#         cv2.imshow("Image", img)
        
        if label == "p_number":
            if is_valid_pan(text.strip()):
                result_dict[label] = text.strip()
        elif (label == "name") or (label == "parent"):
            if is_valid_name(text.strip()):
                result_dict[label] = text.strip()
        elif label == "dob":
            if is_valid_dob(text.strip()):
                result_dict[label] = text.strip()
        else:      
            result_dict[label] = text.strip()

    return result_dict
    

def get_prediction(img):
    input_image = np.array(img)

    # Convert the image from BGR to RGB
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Make predictions on the image
    results = model(input_image)

    # Print the results
    prediction  = results.pandas().xyxy[0]
    
    predictions = prediction.sort_values(['class', "confidence"]).drop_duplicates("class",keep="last")
    
    predictions = predictions[predictions['name'].isin(['p_number','name','parent','dob'])]
    
#     results.show()
    return predictions
    
def find_skew_angle(img):
     #convertion to Gray Scale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = canny(gray_img)
    
    # Classic straight-line Hough transform
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    skew = skew_angle_hough_transform(gray_img)
    return skew[0]


