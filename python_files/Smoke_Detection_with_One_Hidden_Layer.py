
# coding: utf-8

# ## 导入TensorFlow等

# In[1]:


import tensorflow as tf
import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# %matplotlib inline


# ## 定义神经网络函数

# In[2]:


def add_layer(inputs, in_size, out_size, num=0, activation_function=None):
    # add one more layer and return the output of this layer
    # 区别：大框架，定义层 layer，里面有 小部件
    
    with tf.name_scope('layer'):
        # 区别：小部件
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights' + str(num))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias' + str(num))
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases, name='We_plus_b' + str(num))
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, name='activation_function' + str(num))
        return outputs


# ## 烟雾检测

# In[3]:


import cv2


# ### 读取数据

# In[4]:


training_set = [np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(1, 401)]
training_set.extend([np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(501, 901)])


# In[5]:


training_set_nparray = np.vstack(training_set)


# In[6]:


training_label = np.array([[1.0 if i < 400 else 0.0] for i in range(800)])


# ### 定义输入输出

# In[7]:


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 256 * 256], name='xs')
    ys = tf.placeholder(tf.float32, [None, 1], name='ys')


# ### 定义隐层和输出层

# In[8]:


layer1 = add_layer(xs, 256 * 256, 1024, 1, activation_function=tf.nn.sigmoid)


# In[9]:


prediction = add_layer(layer1, 1024, 1, 2, activation_function=tf.nn.sigmoid)


# ### 定义损失函数

# In[10]:


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))


# ### 使用全局梯度下降函数

# In[11]:


with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)
# train_step = tf.train.AdamOptimizer().minimize(loss)


# In[12]:


smoke_sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[13]:


writer = tf.summary.FileWriter("logs/", smoke_sess.graph)


# ### 初始化变量

# In[14]:


smoke_sess.run(tf.global_variables_initializer())


# ### 训练模型

# In[15]:


for i in range(1000):
    smoke_sess.run(train_step, feed_dict={xs: training_set_nparray, ys: training_label})
    if i % 50 == 0:
        print(i, smoke_sess.run(loss, feed_dict={xs: training_set_nparray, ys: training_label}))


# ### 在训练集上的结果

# In[ ]:


training_set_prediction = smoke_sess.run(prediction, feed_dict={xs: training_set_nparray})


# In[ ]:


training_set_prediction[training_set_prediction >= 0.5] = 1
training_set_prediction[training_set_prediction < 0.5] = 0


# In[ ]:


training_set_prediction_positive = training_set_prediction[: 400]
training_set_prediction_negative = training_set_prediction[400:]


# In[ ]:


print('FC Training Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(training_set_prediction_positive[training_set_prediction_positive == 1]),len(training_set_prediction_positive[training_set_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(training_set_prediction_negative[training_set_prediction_negative == 0]),len(training_set_prediction_negative[training_set_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(training_set_prediction_positive[training_set_prediction_positive == 1]) + 0.0) / len(training_set_prediction_positive),
    (len(training_set_prediction_negative[training_set_prediction_negative == 1]) + 0.0) / len(training_set_prediction_negative)
))


# ### 在测试集上的结果

# In[ ]:


test_set = [np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(401, 501)]
test_set.extend([np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(901, 1001)])


# In[ ]:


test_set_nparray = np.vstack(test_set)


# In[ ]:


test_set_prediction = smoke_sess.run(prediction, feed_dict={xs: test_set_nparray})


# In[ ]:


test_set_prediction[test_set_prediction >= 0.5] = 1
test_set_prediction[test_set_prediction < 0.5] = 0
test_set_prediction_positive = test_set_prediction[: 100]
test_set_prediction_negative = test_set_prediction[100:]


# In[ ]:


print('FC Test Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(test_set_prediction_positive[test_set_prediction_positive == 1]),len(test_set_prediction_positive[test_set_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(test_set_prediction_negative[test_set_prediction_negative == 0]),len(test_set_prediction_negative[test_set_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(test_set_prediction_positive[test_set_prediction_positive == 1]) + 0.0) / len(test_set_prediction_positive),
    (len(test_set_prediction_negative[test_set_prediction_negative == 1]) + 0.0) / len(test_set_prediction_negative)
))


# ## 提取深度特征结合SVM

# In[ ]:


from sklearn import svm


# In[ ]:


clf = svm.SVC()


# In[ ]:


trainging_set_deep_feature = smoke_sess.run(layer1, feed_dict={xs: training_set_nparray})


# In[ ]:


clf.fit(trainging_set_deep_feature, np.resize(training_label, (800, )))


# In[ ]:


training_set_deep_feature_svm_prediction = clf.predict(trainging_set_deep_feature)


# In[ ]:


training_set_deep_feature_svm_prediction[training_set_deep_feature_svm_prediction >= 0.5] = 1
training_set_deep_feature_svm_prediction[training_set_deep_feature_svm_prediction < 0.5] = 0
training_set_deep_feature_svm_prediction_positive = training_set_deep_feature_svm_prediction[: 400]
training_set_deep_feature_svm_prediction_negative = training_set_deep_feature_svm_prediction[400:]


# In[ ]:


print('FC+SVM Training Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(training_set_deep_feature_svm_prediction_positive[training_set_deep_feature_svm_prediction_positive == 1]),len(training_set_deep_feature_svm_prediction_positive[training_set_deep_feature_svm_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(training_set_deep_feature_svm_prediction_negative[training_set_deep_feature_svm_prediction_negative == 0]),len(training_set_deep_feature_svm_prediction_negative[training_set_deep_feature_svm_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(training_set_deep_feature_svm_prediction_positive[training_set_deep_feature_svm_prediction_positive == 1]) + 0.0) / len(training_set_deep_feature_svm_prediction_positive),
    (len(training_set_deep_feature_svm_prediction_negative[training_set_deep_feature_svm_prediction_negative == 1]) + 0.0) / len(training_set_deep_feature_svm_prediction_negative)
))


# In[ ]:


test_set_deep_feature = smoke_sess.run(layer1, feed_dict={xs: test_set_nparray})


# In[ ]:


test_set_deep_feature_svm_prediction = clf.predict(test_set_deep_feature)


# In[ ]:


test_set_deep_feature_svm_prediction[test_set_deep_feature_svm_prediction >= 0.5] = 1
test_set_deep_feature_svm_prediction[test_set_deep_feature_svm_prediction < 0.5] = 0
test_set_deep_feature_svm_prediction_positive = test_set_deep_feature_svm_prediction[: 100]
test_set_deep_feature_svm_prediction_negative = test_set_deep_feature_svm_prediction[100:]


# In[ ]:


print('FC+SVM Test Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(test_set_deep_feature_svm_prediction_positive[test_set_deep_feature_svm_prediction_positive == 1]),len(test_set_deep_feature_svm_prediction_positive[test_set_deep_feature_svm_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(test_set_deep_feature_svm_prediction_negative[test_set_deep_feature_svm_prediction_negative == 0]),len(test_set_deep_feature_svm_prediction_negative[test_set_deep_feature_svm_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(test_set_deep_feature_svm_prediction_positive[test_set_deep_feature_svm_prediction_positive == 1]) + 0.0) / len(test_set_deep_feature_svm_prediction_positive),
    (len(test_set_deep_feature_svm_prediction_negative[test_set_deep_feature_svm_prediction_negative == 1]) + 0.0) / len(test_set_deep_feature_svm_prediction_negative)
))

