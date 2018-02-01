
# coding: utf-8

# # 导入相关库

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.svm import SVC


# In[3]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


# # 读取VGG16预训练模型参数

# In[4]:


model = VGG16(weights='imagenet', include_top=False)


# # 读取训练集图片计算特征

# In[5]:


training_set_vgg16_features = []
for i in list(range(1, 401)) + list(range(501, 901)):
    img_path = 'test8/%06d.png' % i
    # print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    training_set_vgg16_features.append(model.predict(x).reshape((7*7*512, )))


# In[6]:


training_set_vgg16_features_ndarray = np.vstack(training_set_vgg16_features)


# In[7]:


training_set_label = np.array([1.0 if i < 400 else 0.0 for i in range(800)])


# # 读取测试集图片计算特征

# In[8]:


test_set_vgg16_features = []
for i in list(range(401, 501)) + list(range(901, 1001)):
    img_path = 'test8/%06d.png' % i
    # print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    test_set_vgg16_features.append(model.predict(x).reshape((7*7*512, )))


# In[9]:


test_set_vgg16_features_ndarray = np.vstack(test_set_vgg16_features)


# # VGG16+SVM

# ### 定义SVM

# In[ ]:


clf_vgg = SVC()


# ### 训练SVM

# In[ ]:


clf_vgg.fit(training_set_vgg16_features_ndarray, training_set_label)


# ### 计算SVM在训练集上的预测值

# In[ ]:


training_set_prediction = clf_vgg.predict(training_set_vgg16_features_ndarray)


# In[ ]:


training_set_prediction[training_set_prediction >= 0.5] = 1
training_set_prediction[training_set_prediction < 0.5] = 0


# In[ ]:


training_set_prediction_positive = training_set_prediction[: 400]
training_set_prediction_negative = training_set_prediction[400:]


# ### 计算训练集准确率

# In[ ]:


print('VGG16+SVM，训练集\n检验率： {0}, 虚警率： {1}'.format(len(training_set_prediction_positive[training_set_prediction_positive == 1]) / len(training_set_prediction_positive), len(training_set_prediction_negative[training_set_prediction_negative == 1]) / len(training_set_prediction_negative)))


# ### 计算SVM在测试集上的预测值

# In[ ]:


test_set_prediction = clf_vgg.predict(test_set_vgg16_features_ndarray)


# In[ ]:


test_set_prediction[test_set_prediction >= 0.5] = 1
test_set_prediction[test_set_prediction < 0.5] = 0
test_set_prediction_positive = test_set_prediction[: 100]
test_set_prediction_negative = test_set_prediction[100:]


# ### 计算测试集准确率

# In[ ]:


print('VGG16+SVM，测试集\n检验率： {0}, 虚警率： {1}'.format(len(test_set_prediction_positive[test_set_prediction_positive == 1]) / len(test_set_prediction_positive), len(test_set_prediction_negative[test_set_prediction_negative == 1]) / len(test_set_prediction_negative)))


# # 全连接神经网络
