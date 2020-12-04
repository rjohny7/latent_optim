#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
from PIL import Image
from tqdm import tqdm


# In[3]:



data_dir = "C:\\Users\\Andrea\\Downloads\\danbooru-images\\" 
faceCascade = cv2.CascadeClassifier('C:\\Users\\Andrea\\Documents\\Fall2020\\EE460J\\FinalProject\\lbpcascade_animeface.xml')
output_dir = "C:\\Users\\Andrea\\Documents\\Fall2020\\EE460J\\FinalProject\\cropped\\"
file_name = "db"
crop_size = (256,256)
only_color = True

def biggest_rectangle(r):
    #return w*h
    return r[2]*r[3]

count = 1
for c, filename in enumerate(tqdm(os.listdir(data_dir))):
    image = cv2.imread(data_dir+filename)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        gray = cv2.equalizeHist(gray)
        # detector options
        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.01, minNeighbors = 5, minSize = (90, 90))
        #if any faces are detected, we only extract the biggest detected region
        if len(faces) == 0:
            continue
        elif len(faces) > 1:
            sorted(faces, key=biggest_rectangle, reverse=True)

        if only_color and (Image.fromarray(image).convert('RGB').getcolors() is not None):
            continue

        x, y, w, h = faces[0]
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_image = image[y:y + h, x:x + w,:]
        resized_image = cv2.resize(cropped_image, crop_size)
        cv2.imwrite(output_dir+str(count+6593)+file_name+".png", resized_image)
        count= count +1

