
# coding: utf-8

# ## 导入cv2等库 读入图片

# In[ ]:


import cv2
import numpy as np
from sklearn import svm, preprocessing
from skimage import feature as ft


# In[ ]:


training_set = []
for i in range(1, 401):
    training_set.append(cv2.imread('../test8/%06d.png' % i))
for i in range(502, 902):
    training_set.append(cv2.imread('../test8/%06d.png' % i))


# In[ ]:


training_label = np.array([1 if i < 400 else 0 for i in range(800)])


# In[ ]:


test_set = []
for i in range(401, 502):
    test_set.append(cv2.imread('../test8/%06d.png' % i))
for i in range(902, 1001):
    test_set.append(cv2.imread('../test8/%06d.png' % i))


# ## 计算HOG特征向量

# In[ ]:


training_set_hog = [ft.hog(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (256, 256))) for i in training_set]
training_set_hog_nparray = np.vstack(training_set_hog)


# In[ ]:


test_set_hog = [ft.hog(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (256, 256))) for i in test_set]
test_set_hog_nparray = np.vstack(test_set_hog)


# ## 训练svm

# In[ ]:


clf_hog = svm.SVC()


# In[ ]:


clf_hog.fit(training_set_hog_nparray, training_label)


# In[ ]:


test_predict_label_hog = clf_hog.predict(test_set_hog_nparray)


# In[ ]:


test_predict_label_hog_positive = test_predict_label_hog[:100]
test_predict_label_hog_negative = test_predict_label_hog[100:]


# In[ ]:


print '检测率: {0}, 虚警率: {1}'.format(
    (len(test_predict_label_hog_positive[test_predict_label_hog_positive == 1]) + 0.0) / len(test_predict_label_hog_positive),
    (len(test_predict_label_hog_negative[test_predict_label_hog_negative == 1]) + 0.0) / len(test_predict_label_hog_negative)
)


# ## 计算LBP特征向量

# In[ ]:


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


# In[ ]:


lbp_descriptor = LocalBinaryPatterns(8 * 8, 8)


# In[ ]:


training_set_lbp = [lbp_descriptor.describe(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (256, 256))) for i in training_set]
training_set_lbp_nparray = np.vstack(training_set_lbp)


# In[ ]:


test_set_lbp = [lbp_descriptor.describe(cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (256, 256))) for i in test_set]
test_set_lbp_nparray = np.vstack(test_set_lbp)


# ## 训练svm

# In[ ]:


clf_lbp = svm.SVC()


# In[ ]:


clf_lbp.fit(training_set_lbp_nparray, training_label)


# In[ ]:


test_predict_label_lbp = clf_lbp.predict(test_set_lbp_nparray)


# In[ ]:


test_predict_label_lbp_positive = test_predict_label_lbp[:100]
test_predict_label_lbp_negative = test_predict_label_lbp[100:]


# In[ ]:


print '检测率: {0}, 虚警率: {1}'.format(
    (len(test_predict_label_lbp_positive[test_predict_label_lbp_positive == 1]) + 0.0) / len(test_predict_label_lbp_positive),
    (len(test_predict_label_lbp_negative[test_predict_label_lbp_negative == 1]) + 0.0) / len(test_predict_label_lbp_negative)
)


# ## 爬取图片

# In[ ]:


from bs4 import BeautifulSoup as bs
import requests


# In[ ]:


import re  
import requests  

def dowmloadPic(html,keyword , i ):  
    pic_url = re.findall('"objURL":"(.*?)",',html,re.S)     
    print '找到关键词:'+keyword+'的图片，现在开始下载图片...'  
    for each in pic_url:  
        print u'正在下载第'+str(i+1)+u'张图片，图片地址:'+str(each)  
        try:  
            pic= requests.get(each, timeout=50)  
        except  Exception,ex :  
            print u'【错误】当前图片无法下载'   
            continue  
        string = 'pictures\\'+keyword+'_'+str(i) + '.jpg'  
        #resolve the problem of encode, make sure that chinese name could be store  
        fp = open(string.decode('utf-8').encode('cp936'),'wb')  
        fp.write(pic.content)  
        fp.close()  
        i += 1  
    return i  
  
if __name__ == '__main__':  
    word =  raw_input('Input keywords:')  
    word = word.decode('cp936').encode('utf-8')  
    #url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&ct=201326592&v=flip'  
    pnMax = input('Input max pn:')  
    pncount = 0  
    gsm = 80  #这个值不知干嘛的  
    str_gsm =str(gsm)  
    if not os.path.exists('pictures'):  
        os.mkdir('pictures')  
    while pncount<pnMax:  
        str_pn = str(pncount)  
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&pn='+str_pn+'&gsm='+str_gsm+'&ct=&ic=0&lm=-1&width=0&height=0'  
        result = requests.get(url)  
        pncount = dowmloadPic(result.text,word ,pncount)  
    print u'下载完毕'  


# ## 绘制示意图

# In[ ]:


import cv2
import numpy as np
from skimage import feature as ft
from skimage import img_as_float


# ### 读取图片

# In[ ]:


img = cv2.imread('../test8/000001.png')


# ### 计算灰度图

# In[ ]:


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ### 显示灰度图

# In[ ]:


cv2.imshow('', img_gray)
cv2.waitKey(5000)


# ### 计算hog特征和hog图

# In[ ]:


vec_hog, img_hog = ft.hog(img_gray, visualise=True)


# ### 显示图片

# In[ ]:


cv2.imshow('', img_hog)
cv2.waitKey(5000)


# ### 计算LBP图

# In[ ]:


img_lbp = ft.local_binary_pattern(img_gray, 8 * 8, 8)


# In[ ]:


cv2.imshow('', img_lbp)
cv2.waitKey(5000)


# In[ ]:


cv2.imshow('', np.hstack([img_as_float(img_gray), img_hog, img_lbp]))
cv2.waitKey(0)

