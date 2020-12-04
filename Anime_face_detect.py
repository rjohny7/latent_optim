#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
from PIL import Image
from tqdm import tqdm

raw_data_dir = "D:\\william\\UT Courses\\2020Fall\\EE460J\\Lab5\\GAN_Dataset\\dataset\\"
data_dir = "D:\\william\\UT Courses\\2020Fall\\EE460J\\Lab5\\GAN_Dataset\\dataset_chara_2\\"
faceCascade = cv2.CascadeClassifier('D:\\william\\UT Courses\\2020Fall\\EE460J\\Lab5\\GAN_Dataset\\lbpcascade_animeface.xml')
output_dir = "D:\\william\\UT Courses\\2020Fall\\EE460J\\Lab5\\GAN_Dataset\\getchu_dataset_256_2\\"
file_name = "gc_"
crop_size = (256,256)
only_color = True

def biggest_rectangle(r):
    #return w*h
    return r[2]*r[3]

for dir1 in os.listdir(raw_data_dir):
    #if os.path.isdir(data_dir+dir1+"\\"):
    if not '.' in dir1:
        cur_dir1 = data_dir+dir1+"\\"
        for dir2 in os.listdir(cur_dir1):
            if not '.' in dir2:
                cur_dir2 = cur_dir1+dir2+"\\"
                
                for filename in os.listdir(cur_dir2):
                    if (filename.endswith('.jpg')) and ("chara" in filename):
                        img = cv2.imread(cur_dir2+filename)
                        if img is not None and img.shape[:2] != (1,1):
                            copyfile(cur_dir2+filename, data_dir+filename[:-4]+".png")
                            count += 1

for count,filename in enumerate(tqdm(os.listdir(data_dir))):
    image = cv2.imread(data_dir+filename)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        gray = cv2.equalizeHist(gray)
        # detector options
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor = 1.01,
                                             minNeighbors = 5,
                                             minSize = (90, 90))
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
        cv2.imwrite(output_dir+file_name+str(count)+".png", resized_image)


# In[ ]:




