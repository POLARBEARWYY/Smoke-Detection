
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from skimage import feature as ft
from skimage import img_as_float


# In[ ]:


img = cv2.imread('test8/000001.png')


# In[ ]:


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[ ]:


vec_hog, img_hog = ft.hog(img_gray, visualise=True)


# In[ ]:


img_lbp = ft.local_binary_pattern(img_gray, 8 * 8, 8)


# In[ ]:


cv2.imshow('', np.hstack([img_as_float(img_gray), img_hog, img_lbp]))
cv2.waitKey(0)

