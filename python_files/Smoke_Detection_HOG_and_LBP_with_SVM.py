
# coding: utf-8

# ## 导入cv2等库 读入图片

# In[1]:


import cv2
import numpy as np
from sklearn import svm, preprocessing
from skimage import feature as ft


# In[2]:


training_set = []
for i in range(1, 401):
    training_set.append(cv2.imread('test8/%06d.png' % i))
for i in range(502, 902):
    training_set.append(cv2.imread('test8/%06d.png' % i))


# In[22]:


training_label = np.array([1 if i < 400 else 0 for i in range(800)])


# In[24]:


test_set = []
for i in range(401, 502):
    test_set.append(cv2.imread('test8/%06d.png' % i))
for i in range(902, 1001):
    test_set.append(cv2.imread('test8/%06d.png' % i))


# ## 计算HOG特征向量

# In[25]:


training_set_hog = [ft.hog(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (224, 224))) for i in training_set]
training_set_hog_nparray = np.vstack(training_set_hog)


# In[30]:


test_set_hog = [ft.hog(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (224, 224))) for i in test_set]
test_set_hog_nparray = np.vstack(test_set_hog)


# ## 训练svm

# In[27]:


clf_hog = svm.SVC()


# In[28]:


clf_hog.fit(training_set_hog_nparray, training_label)


# In[31]:


test_predict_label_hog = clf_hog.predict(test_set_hog_nparray)


# In[69]:


test_predict_label_hog_positive = test_predict_label_hog[:100]
test_predict_label_hog_negative = test_predict_label_hog[100:]


# In[11]:


training_predict_label_hog = clf_hog.predict(training_set_hog_nparray)
training_predict_label_hog_positive = training_predict_label_hog[:400]
training_predict_label_hog_negative = training_predict_label_hog[400:]


# In[24]:


print('HOG+SVM Training Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(training_predict_label_hog_positive[training_predict_label_hog_positive == 1]),len(training_predict_label_hog_positive[training_predict_label_hog_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(training_predict_label_hog_negative[training_predict_label_hog_negative == 0]),len(training_predict_label_hog_negative[training_predict_label_hog_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(training_predict_label_hog_positive[training_predict_label_hog_positive == 1]) + 0.0) / len(training_predict_label_hog_positive),
    (len(training_predict_label_hog_negative[training_predict_label_hog_negative == 1]) + 0.0) / len(training_predict_label_hog_negative)
))


# In[70]:


print('HOG+SVM Test Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(test_predict_label_hog_positive[test_predict_label_hog_positive == 1]),len(test_predict_label_hog_positive[test_predict_label_hog_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(test_predict_label_hog_negative[test_predict_label_hog_negative == 0]),len(test_predict_label_hog_negative[test_predict_label_hog_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(test_predict_label_hog_positive[test_predict_label_hog_positive == 1]) + 0.0) / len(test_predict_label_hog_positive),
    (len(test_predict_label_hog_negative[test_predict_label_hog_negative == 1]) + 0.0) / len(test_predict_label_hog_negative)
))


# ## 计算LBP特征向量

# In[13]:


class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = ft.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist


# In[14]:


lbp_descriptor = LocalBinaryPatterns(16, 2)


# In[15]:


training_set_lbp = [lbp_descriptor.describe(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (256, 256))) for i in training_set]
training_set_lbp_nparray = np.vstack(training_set_lbp)


# In[16]:


test_set_lbp = [lbp_descriptor.describe(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (256, 256))) for i in test_set]
test_set_lbp_nparray = np.vstack(test_set_lbp)


# ## 训练svm

# In[17]:


clf_lbp = svm.SVC()


# In[18]:


clf_lbp.fit(training_set_lbp_nparray, training_label)


# In[19]:


test_predict_label_lbp = clf_lbp.predict(test_set_lbp_nparray)


# In[20]:


test_predict_label_lbp_positive = test_predict_label_lbp[:100]
test_predict_label_lbp_negative = test_predict_label_lbp[100:]


# In[21]:


training_predict_label_lbp = clf_lbp.predict(training_set_lbp_nparray)
training_predict_label_lbp_positive = training_predict_label_lbp[:400]
training_predict_label_lbp_negative = training_predict_label_lbp[400:]


# In[23]:


print('LBP+SVM Training Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(training_predict_label_lbp_positive[training_predict_label_lbp_positive == 1]),len(training_predict_label_lbp_positive[training_predict_label_lbp_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(training_predict_label_lbp_negative[training_predict_label_lbp_negative == 0]),len(training_predict_label_lbp_negative[training_predict_label_lbp_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(training_predict_label_lbp_positive[training_predict_label_lbp_positive == 1]) + 0.0) / len(training_predict_label_lbp_positive),
    (len(training_predict_label_lbp_negative[training_predict_label_lbp_negative == 1]) + 0.0) / len(training_predict_label_lbp_negative)
))


# In[22]:


print('LBP+SVM Test Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(test_predict_label_lbp_positive[test_predict_label_lbp_positive == 1]),len(test_predict_label_lbp_positive[test_predict_label_lbp_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(test_predict_label_lbp_negative[test_predict_label_lbp_negative == 0]),len(test_predict_label_lbp_negative[test_predict_label_lbp_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(test_predict_label_lbp_positive[test_predict_label_lbp_positive == 1]) + 0.0) / len(test_predict_label_lbp_positive),
    (len(test_predict_label_lbp_negative[test_predict_label_lbp_negative == 1]) + 0.0) / len(test_predict_label_lbp_negative)
))

