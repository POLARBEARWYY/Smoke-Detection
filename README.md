# Smoke-Detection For Alysa
A smoke detector based on image and video.  
Early warning of fire disaster is one of the important usage of smoke detection.
Smoke detector which based on video or image can avoid interference of environmental especially in open or lager spaces and outdoor environments.Generally speaking, video based smoke detection methods distinguish smoke from non-smoke objects based on some distinctive features such as motion, edge, color and texture.
***
## Requirements
- Python3
- numpy
- sklearn
- opencv
- TensorFlow
- Keras
***
## 各个文件详解
（英文实在编不下去了下面就用中文了）
#### Fire_Detection_HOG_and_LBP_with_SVM
使用Opencv读取图片转换为灰度图，再使用skimage提取HOG和LBP特征，再使用sklearn中的SVM类判断是否有烟雾
#### Fire_Detection_with_One_Hidden_Layer
使用TensorFlow建立一个具有单隐层的全连接神经网络，将图片转化为灰度图后压缩为1维数组作为输入，输出为0/1即是否有烟雾，训练完成后提取隐层输出作为图片特征，再使用SVM训练判断是否有烟雾，即将神经网络作为特征提取器
#### Fire_Detection_VGG16_with_SVM_and_Fully_Connected_Network
使用Keras读取已经在ImageNet上训练好的VGG16模型参数，将VGG16作为特征提取器得到VGG16最后一个池化层的输出（即“bottleneck feature”）作为图片特征，再定义一个具有两个隐层的全连接神经网络，最终输出为0/1即是否有烟雾（为了防止过拟合其中加入Dropout）
#### show_hog_and_lbp
显示HOG/LBP特征图
#### spider001
从百度图片爬取图片作为训练数据
