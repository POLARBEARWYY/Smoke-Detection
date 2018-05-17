
# coding: utf-8

# In[1]:


import cv2
import numpy as np

from keras.models import Model
from keras.layers import Activation, Dropout, Dense, Input, Flatten


# # load data

# In[2]:


smoke_dataset_image_1 = np.load('smoke_data/set1_48_data.npy')
smoke_dataset_label_1 = np.load('smoke_data/set1_48_label.npy')
smoke_dataset_image_2 = np.load('smoke_data/set2_48_data.npy')
smoke_dataset_label_2 = np.load('smoke_data/set2_48_label.npy')
smoke_dataset_image_3 = np.load('smoke_data/set3_48_data.npy')
smoke_dataset_label_3 = np.load('smoke_data/set3_48_label.npy')
smoke_dataset_image_4 = np.load('smoke_data/set4_48_data.npy')
smoke_dataset_label_4 = np.load('smoke_data/set4_48_label.npy')


# In[3]:


training_set = smoke_dataset_image_3 * 255
training_set_label = smoke_dataset_label_3.flatten().astype(np.int64)
test_set = smoke_dataset_image_4 * 255
test_set_label = smoke_dataset_label_4.flatten().astype(np.int64)


# # HOG

# ## Get Feature

# In[4]:


from skimage import feature as ft


# In[5]:


training_set_hog_feature = np.array([ft.hog(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)) for i in training_set])

test_set_hog_feature = np.array([ft.hog(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)) for i in test_set])


# ## Construct Fully Connected Network

# In[6]:


smoke_detector_hog_input = Input(shape=(training_set_hog_feature.shape[1], ))


# In[7]:


smoke_detector_hog_dense = Dense(128, activation='sigmoid')(smoke_detector_hog_input)


# In[8]:


smoke_detector_hog_dropout = Dropout(0.3)(smoke_detector_hog_dense)


# In[9]:


smoke_detector_hog_prediction = Dense(1, activation='sigmoid')(smoke_detector_hog_dropout)


# In[10]:


smoke_detector_hog_model = Model(inputs=smoke_detector_hog_input, outputs=smoke_detector_hog_prediction)


# In[11]:


smoke_detector_hog_model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[39]:


smoke_detector_hog_model.fit(training_set_hog_feature, training_set_label, epochs=30)


# ## Test HOG Model

# In[40]:


smoke_detector_hog_model.evaluate(training_set_hog_feature, training_set_label)


# In[41]:


training_set_label_prediction_hog = smoke_detector_hog_model.predict(training_set_hog_feature)
test_set_label_prediction_hog = smoke_detector_hog_model.predict(test_set_hog_feature)


# In[42]:


training_set_label_prediction_hog_positive = training_set_label_prediction_hog[training_set_label == 1]
training_set_label_prediction_hog_negative = training_set_label_prediction_hog[training_set_label == 0]

training_set_label_prediction_hog_positive_true = training_set_label_prediction_hog_positive[training_set_label_prediction_hog_positive >= 0.5]
training_set_label_prediction_hog_positive_false = training_set_label_prediction_hog_positive[training_set_label_prediction_hog_positive < 0.5]
training_set_label_prediction_hog_negative_true = training_set_label_prediction_hog_negative[training_set_label_prediction_hog_negative < 0.5]
training_set_label_prediction_hog_negative_false = training_set_label_prediction_hog_negative[training_set_label_prediction_hog_negative >= 0.5]

print(
    '''HOG+FC Training Set 混淆矩阵 Training Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    training_set_label_prediction_hog_positive_true.shape[0],
    training_set_label_prediction_hog_positive_false.shape[0],
    training_set_label_prediction_hog_negative_true.shape[0],
    training_set_label_prediction_hog_negative_false.shape[0],
    training_set_label_prediction_hog_positive_true.shape[0] / training_set_label_prediction_hog_positive.shape[0],
    training_set_label_prediction_hog_negative_false.shape[0] / training_set_label_prediction_hog_negative.shape[0]
    )
)


# In[43]:


test_set_label_prediction_hog_positive = test_set_label_prediction_hog[test_set_label == 1]
test_set_label_prediction_hog_negative = test_set_label_prediction_hog[test_set_label == 0]

test_set_label_prediction_hog_positive_true = test_set_label_prediction_hog_positive[test_set_label_prediction_hog_positive >= 0.5]
test_set_label_prediction_hog_positive_false = test_set_label_prediction_hog_positive[test_set_label_prediction_hog_positive < 0.5]
test_set_label_prediction_hog_negative_true = test_set_label_prediction_hog_negative[test_set_label_prediction_hog_negative < 0.5]
test_set_label_prediction_hog_negative_false = test_set_label_prediction_hog_negative[test_set_label_prediction_hog_negative >= 0.5]

print(
    '''HOG+FC Test Set 混淆矩阵 test Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    test_set_label_prediction_hog_positive_true.shape[0],
    test_set_label_prediction_hog_positive_false.shape[0],
    test_set_label_prediction_hog_negative_true.shape[0],
    test_set_label_prediction_hog_negative_false.shape[0],
    test_set_label_prediction_hog_positive_true.shape[0] / test_set_label_prediction_hog_positive.shape[0],
    test_set_label_prediction_hog_negative_false.shape[0] / test_set_label_prediction_hog_negative.shape[0]
    )
)


# ## Extract Feature

# In[12]:


smoke_detector_hog_model_feature_extraction = Model(inputs=smoke_detector_hog_input, outputs=smoke_detector_hog_dense)


# In[45]:


smoke_detector_hog_model_feature_extraction.predict(training_set_hog_feature).shape


# # LBP

# ## Get Feature

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


training_set_lbp_feature = np.array([lbp_descriptor.describe(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)) for i in training_set])

test_set_lbp_feature = np.array([lbp_descriptor.describe(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)) for i in test_set])


# In[16]:


smoke_detector_lbp_input = Input(shape=(training_set_lbp_feature.shape[1], ))


# In[17]:


smoke_detector_lbp_dense = Dense(128, activation='sigmoid')(smoke_detector_lbp_input)


# In[18]:


smoke_detector_lbp_dropout = Dropout(0.3)(smoke_detector_lbp_dense)


# In[19]:


smoke_detector_lbp_prediction = Dense(1, activation='sigmoid')(smoke_detector_lbp_dropout)


# In[20]:


smoke_detector_lbp_model = Model(inputs=smoke_detector_lbp_input, outputs=smoke_detector_lbp_prediction)


# In[21]:


smoke_detector_lbp_model.compile(optimizer='AdaDelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[55]:


smoke_detector_lbp_model.fit(training_set_lbp_feature, training_set_label, epochs=30)


# ## Test LBP Model

# In[56]:


smoke_detector_lbp_model.evaluate(training_set_lbp_feature, training_set_label)


# In[57]:


training_set_label_prediction_lbp = smoke_detector_lbp_model.predict(training_set_lbp_feature)
test_set_label_prediction_lbp = smoke_detector_lbp_model.predict(test_set_lbp_feature)


# In[58]:


training_set_label_prediction_lbp_positive = training_set_label_prediction_lbp[training_set_label == 1]
training_set_label_prediction_lbp_negative = training_set_label_prediction_lbp[training_set_label == 0]

training_set_label_prediction_lbp_positive_true = training_set_label_prediction_lbp_positive[training_set_label_prediction_lbp_positive >= 0.5]
training_set_label_prediction_lbp_positive_false = training_set_label_prediction_lbp_positive[training_set_label_prediction_lbp_positive < 0.5]
training_set_label_prediction_lbp_negative_true = training_set_label_prediction_lbp_negative[training_set_label_prediction_lbp_negative < 0.5]
training_set_label_prediction_lbp_negative_false = training_set_label_prediction_lbp_negative[training_set_label_prediction_lbp_negative >= 0.5]

print(
    '''LBP+FC Training Set 混淆矩阵 Training Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    training_set_label_prediction_lbp_positive_true.shape[0],
    training_set_label_prediction_lbp_positive_false.shape[0],
    training_set_label_prediction_lbp_negative_true.shape[0],
    training_set_label_prediction_lbp_negative_false.shape[0],
    training_set_label_prediction_lbp_positive_true.shape[0] / training_set_label_prediction_lbp_positive.shape[0],
    training_set_label_prediction_lbp_negative_false.shape[0] / training_set_label_prediction_lbp_negative.shape[0]
    )
)


# In[59]:


test_set_label_prediction_lbp_positive = test_set_label_prediction_lbp[test_set_label == 1]
test_set_label_prediction_lbp_negative = test_set_label_prediction_lbp[test_set_label == 0]

test_set_label_prediction_lbp_positive_true = test_set_label_prediction_lbp_positive[test_set_label_prediction_lbp_positive >= 0.5]
test_set_label_prediction_lbp_positive_false = test_set_label_prediction_lbp_positive[test_set_label_prediction_lbp_positive < 0.5]
test_set_label_prediction_lbp_negative_true = test_set_label_prediction_lbp_negative[test_set_label_prediction_lbp_negative < 0.5]
test_set_label_prediction_lbp_negative_false = test_set_label_prediction_lbp_negative[test_set_label_prediction_lbp_negative >= 0.5]

print(
    '''LBP+FC Test Set 混淆矩阵 test Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    test_set_label_prediction_lbp_positive_true.shape[0],
    test_set_label_prediction_lbp_positive_false.shape[0],
    test_set_label_prediction_lbp_negative_true.shape[0],
    test_set_label_prediction_lbp_negative_false.shape[0],
    test_set_label_prediction_lbp_positive_true.shape[0] / test_set_label_prediction_lbp_positive.shape[0],
    test_set_label_prediction_lbp_negative_false.shape[0] / test_set_label_prediction_lbp_negative.shape[0]
    )
)


# ## Extract Feature

# In[22]:


smoke_detector_lbp_model_feature_extraction = Model(inputs=smoke_detector_lbp_input, outputs=smoke_detector_lbp_dense)


# In[61]:


smoke_detector_lbp_model_feature_extraction.predict(training_set_lbp_feature).shape


# # VGG16

# ## Get Feature

# In[4]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


# In[5]:


model_vgg16 = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')


# In[7]:


training_set_224 = np.array([cv2.resize(i, (224, 224)) for i in smoke_dataset_image_3]) * 255
test_set_224 = np.array([cv2.resize(i, (224, 224)) for i in smoke_dataset_image_4]) * 255


# In[10]:


training_set_vgg16_feature = model_vgg16.predict(preprocess_input(training_set_224))
#test_set_vgg16_feature = model_vgg16.predict(preprocess_input(test_set_224))


# In[23]:


smoke_detector_vgg16_input = Input(shape=(7, 7, 512,))


# In[24]:


smoke_detector_vgg16_flatten = Flatten()(smoke_detector_vgg16_input)


# In[25]:


smoke_detector_vgg16_dense_1 = Dense(1024, activation='sigmoid')(smoke_detector_vgg16_flatten)


# In[26]:


smoke_detector_vgg16_dropout_1 = Dropout(0.3)(smoke_detector_vgg16_dense_1)


# In[27]:


smoke_detector_vgg16_dense_2 = Dense(128, activation='sigmoid')(smoke_detector_vgg16_dropout_1)


# In[28]:


smoke_detector_vgg16_dropout_2 = Dropout(0.3)(smoke_detector_vgg16_dense_2)


# In[29]:


smoke_detector_vgg16_prediction = Dense(1, activation='sigmoid')(smoke_detector_vgg16_dropout_2)


# In[30]:


smoke_detector_vgg16_model = Model(inputs=smoke_detector_vgg16_input, outputs=smoke_detector_vgg16_prediction)


# In[31]:


smoke_detector_vgg16_model.compile(optimizer='AdaDelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[22]:


smoke_detector_vgg16_model.fit(training_set_vgg16_feature, training_set_label, epochs=10)


# ## Test VGG16 Model

# In[23]:


test_set_vgg16_feature = model_vgg16.predict(preprocess_input(test_set_224))


# In[24]:


training_set_label_prediction_vgg16 = smoke_detector_vgg16_model.predict(training_set_vgg16_feature)
test_set_label_prediction_vgg16 = smoke_detector_vgg16_model.predict(test_set_vgg16_feature)


# In[25]:


training_set_label_prediction_vgg16_positive = training_set_label_prediction_vgg16[training_set_label == 1]
training_set_label_prediction_vgg16_negative = training_set_label_prediction_vgg16[training_set_label == 0]

training_set_label_prediction_vgg16_positive_true = training_set_label_prediction_vgg16_positive[training_set_label_prediction_vgg16_positive >= 0.5]
training_set_label_prediction_vgg16_positive_false = training_set_label_prediction_vgg16_positive[training_set_label_prediction_vgg16_positive < 0.5]
training_set_label_prediction_vgg16_negative_true = training_set_label_prediction_vgg16_negative[training_set_label_prediction_vgg16_negative < 0.5]
training_set_label_prediction_vgg16_negative_false = training_set_label_prediction_vgg16_negative[training_set_label_prediction_vgg16_negative >= 0.5]

print(
    '''VGG16+FC Training Set 混淆矩阵 Training Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    training_set_label_prediction_vgg16_positive_true.shape[0],
    training_set_label_prediction_vgg16_positive_false.shape[0],
    training_set_label_prediction_vgg16_negative_true.shape[0],
    training_set_label_prediction_vgg16_negative_false.shape[0],
    training_set_label_prediction_vgg16_positive_true.shape[0] / training_set_label_prediction_vgg16_positive.shape[0],
    training_set_label_prediction_vgg16_negative_false.shape[0] / training_set_label_prediction_vgg16_negative.shape[0]
    )
)


# In[26]:


test_set_label_prediction_vgg16_positive = test_set_label_prediction_vgg16[test_set_label == 1]
test_set_label_prediction_vgg16_negative = test_set_label_prediction_vgg16[test_set_label == 0]

test_set_label_prediction_vgg16_positive_true = test_set_label_prediction_vgg16_positive[test_set_label_prediction_vgg16_positive >= 0.5]
test_set_label_prediction_vgg16_positive_false = test_set_label_prediction_vgg16_positive[test_set_label_prediction_vgg16_positive < 0.5]
test_set_label_prediction_vgg16_negative_true = test_set_label_prediction_vgg16_negative[test_set_label_prediction_vgg16_negative < 0.5]
test_set_label_prediction_vgg16_negative_false = test_set_label_prediction_vgg16_negative[test_set_label_prediction_vgg16_negative >= 0.5]

print(
    '''VGG16+FC Test Set 混淆矩阵 test Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    test_set_label_prediction_vgg16_positive_true.shape[0],
    test_set_label_prediction_vgg16_positive_false.shape[0],
    test_set_label_prediction_vgg16_negative_true.shape[0],
    test_set_label_prediction_vgg16_negative_false.shape[0],
    test_set_label_prediction_vgg16_positive_true.shape[0] / test_set_label_prediction_vgg16_positive.shape[0],
    test_set_label_prediction_vgg16_negative_false.shape[0] / test_set_label_prediction_vgg16_negative.shape[0]
    )
)


# ## Extract Feature

# In[32]:


smoke_detector_vgg16_model_feature_extraction = Model(inputs=smoke_detector_vgg16_input, outputs=smoke_detector_vgg16_dense_2)


# In[29]:


smoke_detector_vgg16_model_feature_extraction.predict(training_set_vgg16_feature).shape


# # Feature Fusion

# In[33]:


from keras.layers import concatenate


# In[34]:


smoke_detector_feature_fusion_concat = concatenate([smoke_detector_hog_model_feature_extraction.output, smoke_detector_lbp_model_feature_extraction.output, smoke_detector_vgg16_model_feature_extraction.output])


# In[35]:


smoke_detector_feature_fusion_dense_1 = Dense(16, activation='sigmoid')(smoke_detector_feature_fusion_concat)


# In[36]:


smoke_detector_feature_fusion_dropout_1 = Dropout(0.2)(smoke_detector_feature_fusion_dense_1)


# In[37]:


smoke_detector_feature_fusion_prediction = Dense(1, activation='sigmoid')(smoke_detector_feature_fusion_dropout_1)


# In[38]:


smoke_detector_feature_fusion_model = Model(inputs=[smoke_detector_hog_input, smoke_detector_lbp_input, smoke_detector_vgg16_input], outputs=[smoke_detector_feature_fusion_prediction])


# In[39]:


smoke_detector_feature_fusion_model.compile(optimizer='AdaDelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[92]:


smoke_detector_feature_fusion_model.fit([training_set_hog_feature, training_set_lbp_feature, training_set_vgg16_feature], training_set_label, epochs=10)


# In[40]:


from keras.utils.vis_utils import plot_model


# In[41]:


plot_model(smoke_detector_feature_fusion_model, to_file='model.png', show_shapes=True)


# In[86]:


smoke_detector_hog_model.save('model/smoke_detector_hog_model.h5')
smoke_detector_lbp_model.save('model/smoke_detector_lbp_model.h5')
smoke_detector_vgg16_model.save('model/smoke_detector_vgg16_model.h5')
smoke_detector_feature_fusion_model.save('model/smoke_detector_feature_fusion_model.h5')


# ## Test Feature Fusion Model

# In[93]:


training_set_label_prediction_feature_fusion = smoke_detector_feature_fusion_model.predict([training_set_hog_feature, training_set_lbp_feature, training_set_vgg16_feature])
test_set_label_prediction_feature_fusion = smoke_detector_feature_fusion_model.predict([test_set_hog_feature, test_set_lbp_feature, test_set_vgg16_feature])


# In[94]:


training_set_label_prediction_feature_fusion_positive = training_set_label_prediction_feature_fusion[training_set_label == 1]
training_set_label_prediction_feature_fusion_negative = training_set_label_prediction_feature_fusion[training_set_label == 0]

training_set_label_prediction_feature_fusion_positive_true = training_set_label_prediction_feature_fusion_positive[training_set_label_prediction_feature_fusion_positive >= 0.5]
training_set_label_prediction_feature_fusion_positive_false = training_set_label_prediction_feature_fusion_positive[training_set_label_prediction_feature_fusion_positive < 0.5]
training_set_label_prediction_feature_fusion_negative_true = training_set_label_prediction_feature_fusion_negative[training_set_label_prediction_feature_fusion_negative < 0.5]
training_set_label_prediction_feature_fusion_negative_false = training_set_label_prediction_feature_fusion_negative[training_set_label_prediction_feature_fusion_negative >= 0.5]

print(
    '''Feature Fusion+FC Training Set 混淆矩阵 Training Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    training_set_label_prediction_feature_fusion_positive_true.shape[0],
    training_set_label_prediction_feature_fusion_positive_false.shape[0],
    training_set_label_prediction_feature_fusion_negative_true.shape[0],
    training_set_label_prediction_feature_fusion_negative_false.shape[0],
    training_set_label_prediction_feature_fusion_positive_true.shape[0] / training_set_label_prediction_feature_fusion_positive.shape[0],
    training_set_label_prediction_feature_fusion_negative_false.shape[0] / training_set_label_prediction_feature_fusion_negative.shape[0]
    )
)


# In[95]:


test_set_label_prediction_feature_fusion_positive = test_set_label_prediction_feature_fusion[test_set_label == 1]
test_set_label_prediction_feature_fusion_negative = test_set_label_prediction_feature_fusion[test_set_label == 0]

test_set_label_prediction_feature_fusion_positive_true = test_set_label_prediction_feature_fusion_positive[test_set_label_prediction_feature_fusion_positive >= 0.5]
test_set_label_prediction_feature_fusion_positive_false = test_set_label_prediction_feature_fusion_positive[test_set_label_prediction_feature_fusion_positive < 0.5]
test_set_label_prediction_feature_fusion_negative_true = test_set_label_prediction_feature_fusion_negative[test_set_label_prediction_feature_fusion_negative < 0.5]
test_set_label_prediction_feature_fusion_negative_false = test_set_label_prediction_feature_fusion_negative[test_set_label_prediction_feature_fusion_negative >= 0.5]

print(
    '''Feature Fusion+FC Test Set 混淆矩阵 test Set
    real/predict   True    False
    
    smoke          {0}     {1}
    no smoke       {2}     {3}
    检测率：{4}
    虚警率：{5}'''.format(
    test_set_label_prediction_feature_fusion_positive_true.shape[0],
    test_set_label_prediction_feature_fusion_positive_false.shape[0],
    test_set_label_prediction_feature_fusion_negative_true.shape[0],
    test_set_label_prediction_feature_fusion_negative_false.shape[0],
    test_set_label_prediction_feature_fusion_positive_true.shape[0] / test_set_label_prediction_feature_fusion_positive.shape[0],
    test_set_label_prediction_feature_fusion_negative_false.shape[0] / test_set_label_prediction_feature_fusion_negative.shape[0]
    )
)

