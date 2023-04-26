#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import numpy as np


# In[3]:





# In[ ]:





# In[4]:




# In[5]:





# In[6]:





# In[7]:





# In[8]:





# In[9]:





# In[10]:




# In[140]:


img = cv2.imread("S15.JPG")
show(img)


# In[141]:


grayscale_Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh_img = cv2.threshold(grayscale_Image, 120, 255, cv2.THRESH_BINARY)
show(thresh_img)


# In[142]:


kernel =np.ones((3),np.uint8)
clear_image = cv2.morphologyEx(thresh_img,cv2.MORPH_OPEN, kernel, iterations=8)
show(clear_image)


# In[143]:


label_image = clear_image.copy()
label_count = 0
rows, cols = label_image.shape
for j in range(rows):
    for i in range(cols):
        pixel = label_image[j, i]
        if 255 == pixel:
            label_count +- 1
            cv.floodFill(label_image, None, (i, j), label_count)


# In[144]:



show(label_image)


# In[145]:


contours, hierarchy = cv.findContours(clear_image,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
output_contour = cv.cvtColor(clear_image, cv.COLOR_GRAY2BGR)
cv.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
print("Number of detected contours", len(contours))


# In[146]:



print(" \n Number of detected contours", len(contours))


# In[147]:


dist_trans = ndimage.distance_transform_edt(clear_image)
local_max = feature.peak_local_max(dist_trans, min_distance=23)
local_max_mask = np.zeros(dist_trans.shape, dtype=bool)
local_max_mask[tuple(local_max.T)] = True
labels = watershed(-dist_trans, measure.label(local_max_mask), mask=clear_image)


# In[148]:


plt.figure(figsize=(10,10))
plt.savefig('foo.png')

fig = plt.figure()
fig
plt.imshow(color.label2rgb(labels, bg_label=0))
plt.imsave('test.png', color.label2rgb(labels, bg_label=0))



os.getcwd()


img = plt.imshow(color.label2rgb(labels, bg_label=0))
type(img)
img.savefig('aaa.png')
from PIL import Image
color.label2rgb(labels, bg_label=0).save()

type(color.label2rgb(labels, bg_label=0))

img = color.label2rgb(labels, bg_label=0)
img.show()

import os
os.getcwd()

print("Number of Wheat grains are : %d" % labels.max())

labels.max()
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
