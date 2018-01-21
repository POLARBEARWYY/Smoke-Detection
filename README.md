# Smoke-Detection For Alysa
A smoke detector based on image and video.  
Early warning of fire disaster is one of the important usage of smoke detection.
Smoke detector which based on video or image can avoid interference of environmental especially in open or lager spaces and outdoor environments.Generally speaking, video based smoke detection methods distinguish smoke from non-smoke objects based on some distinctive features such as motion, edge, color and texture.
## Requirements
- Python3
- numpy
- sklearn
- opencv
- TensorFlow

## 导入TensorFlow等


```python
import tensorflow as tf
import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# %matplotlib inline
```

## 定义神经网络函数


```python
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
```

## 烟雾检测


```python
import cv2
```

### 读取数据


```python
training_set = [np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(1, 401)]
training_set.extend([np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(501, 901)])
```


```python
training_set_nparray = np.vstack(training_set)
```


```python
training_label = np.array([[1.0 if i < 400 else 0.0] for i in range(800)])
```

### 定义输入输出


```python
xs = tf.placeholder(tf.float32, [None, 256 * 256], name='xs')
ys = tf.placeholder(tf.float32, [None, 1], name='ys')
```

### 定义隐层和输出层


```python
layer1 = add_layer(xs, 256 * 256, 1024, 1, activation_function=tf.nn.sigmoid)
```


```python
prediction = add_layer(layer1, 1024, 1, 2, activation_function=tf.nn.sigmoid)
```

### 定义损失函数


```python
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
```

### 使用全局梯度下降函数


```python
train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)
```


```python
smoke_sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)))
```

### 初始化变量


```python
smoke_sess.run(tf.global_variables_initializer())
```

### 训练模型


```python
for i in range(1000):
    smoke_sess.run(train_step, feed_dict={xs: training_set_nparray, ys: training_label})
    if i % 50 == 0:
        print(i, smoke_sess.run(loss, feed_dict={xs: training_set_nparray, ys: training_label}))
```

    0 0.442887
    50 0.361918
    100 0.304387
    150 0.249828
    200 0.222163
    250 0.202497
    300 0.184905
    350 0.180239
    400 0.161046
    450 0.142362
    500 0.142484
    550 0.127693
    600 0.123639
    650 0.142562
    700 0.101554
    750 0.0958758
    800 0.0927265
    850 0.0895231
    900 0.0874223
    950 0.0854852


### 在训练集上的结果


```python
training_set_prediction = smoke_sess.run(prediction, feed_dict={xs: training_set_nparray})
```


```python
training_set_prediction[training_set_prediction >= 0.5] = 1
training_set_prediction[training_set_prediction < 0.5] = 0
```


```python
training_set_prediction_positive = training_set_prediction[: 400]
training_set_prediction_negative = training_set_prediction[400:]
```


```python
print('单隐层神经网络，训练集\n检验率： {0}, 虚警率： {1}'.format(len(training_set_prediction_positive[training_set_prediction_positive == 1]) / len(training_set_prediction_positive), len(training_set_prediction_negative[training_set_prediction_negative == 1]) / len(training_set_prediction_negative)))
```

    单隐层神经网络，训练集
    检验率： 0.935, 虚警率： 0.1


### 在测试集上的结果


```python
test_set = [np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(401, 501)]
test_set.extend([np.reshape(cv2.cvtColor(cv2.resize(cv2.imread('test8/%06d.png' % i), (256, 256)), cv2.COLOR_BGR2GRAY), (1, 256 * 256)) for i in range(901, 1001)])
```


```python
test_set_nparray = np.vstack(test_set)
```


```python
test_set_prediction = smoke_sess.run(prediction, feed_dict={xs: test_set_nparray})
```


```python
test_set_prediction[test_set_prediction >= 0.5] = 1
test_set_prediction[test_set_prediction < 0.5] = 0
test_set_prediction_positive = test_set_prediction[: 100]
test_set_prediction_negative = test_set_prediction[100:]
```


```python
print('单隐层神经网络，测试集\n检验率： {0}, 虚警率： {1}'.format(len(test_set_prediction_positive[test_set_prediction_positive == 1]) / len(test_set_prediction_positive), len(test_set_prediction_negative[test_set_prediction_negative == 1]) / len(test_set_prediction_negative)))
```

    单隐层神经网络，测试集
    检验率： 0.85, 虚警率： 0.21


## 提取深度特征结合SVM


```python
from sklearn import svm
```


```python
clf = svm.SVC()
```


```python
trainging_set_deep_feature = smoke_sess.run(layer1, feed_dict={xs: training_set_nparray})
```


```python
clf.fit(trainging_set_deep_feature, np.resize(training_label, (800, )))
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
training_set_deep_feature_svm_prediction = clf.predict(trainging_set_deep_feature)
```


```python
training_set_deep_feature_svm_prediction[training_set_deep_feature_svm_prediction >= 0.5] = 1
training_set_deep_feature_svm_prediction[training_set_deep_feature_svm_prediction < 0.5] = 0
training_set_deep_feature_svm_prediction_positive = training_set_deep_feature_svm_prediction[: 400]
training_set_deep_feature_svm_prediction_negative = training_set_deep_feature_svm_prediction[400:]
```


```python
print('深度特征+SVM，训练集\n检验率： {0}, 虚警率： {1}'.format(len(training_set_deep_feature_svm_prediction_positive[training_set_deep_feature_svm_prediction_positive == 1]) / len(training_set_deep_feature_svm_prediction_positive), len(training_set_deep_feature_svm_prediction_negative[training_set_deep_feature_svm_prediction_negative == 1]) / len(training_set_deep_feature_svm_prediction_negative)))
```

    深度特征+SVM，训练集
    检验率： 0.9025, 虚警率： 0.1275



```python
test_set_deep_feature = smoke_sess.run(layer1, feed_dict={xs: test_set_nparray})
```


```python
test_set_deep_feature_svm_prediction = clf.predict(test_set_deep_feature)
```


```python
test_set_deep_feature_svm_prediction[test_set_deep_feature_svm_prediction >= 0.5] = 1
test_set_deep_feature_svm_prediction[test_set_deep_feature_svm_prediction < 0.5] = 0
test_set_deep_feature_svm_prediction_positive = test_set_deep_feature_svm_prediction[: 100]
test_set_deep_feature_svm_prediction_negative = test_set_deep_feature_svm_prediction[100:]
```


```python
print('深度特征+SVM，测试集\n检验率： {0}, 虚警率： {1}'.format(len(test_set_deep_feature_svm_prediction_positive[test_set_deep_feature_svm_prediction_positive == 1]) / len(test_set_deep_feature_svm_prediction_positive), len(test_set_deep_feature_svm_prediction_negative[test_set_deep_feature_svm_prediction_negative == 1]) / len(test_set_deep_feature_svm_prediction_negative)))
```

    深度特征+SVM，测试集
    检验率： 0.9, 虚警率： 0.16


## 保存模型


```python
saver = tf.train.Saver()
saver.save(smoke_sess, 'my_test_model')
```




    'my_test_model'



## 使用GPU


```python
def is_gpu_available(cuda_only=True):
    from tensorflow.python.client import device_lib as _device_lib

    if cuda_only:
        return any((x.device_type == 'GPU')
                   for x in _device_lib.list_local_devices())
    else:
        return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
                   for x in _device_lib.list_local_devices())
```


```python
is_gpu_available()
```


```python
def get_available_gpus():
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
```


```python
get_available_gpus()
```


```python
# Creates a graph.
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```
