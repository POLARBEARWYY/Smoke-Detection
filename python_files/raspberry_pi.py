
# coding: utf-8

# In[1]:


import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import cv2
from keras.models import load_model
import time


# In[2]:


model_smoke_detector = load_model('smoke_detecctor_with_mobilenet_48.h5')


# In[3]:


model_mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')


# In[4]:


cap = cv2.VideoCapture(0)
#video_writer = cv2.VideoWriter('smoke_detection_output1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

steps = [5, 150, 500]
#frames = [0] * 30
choice = 0
total_time = 0
count = 0
#num = 1
while True:
    status, img = cap.read()
    #img = img[:, ::-1, :]
    if status:
        #img = img[100: 500, 200: 600, :]
        '''img_detect = cv2.resize(img[:, :, [2, 1, 0]], (48 * num, 48 * num))
        detection_results = []
        for i in range(num):
            for j in range(num):
                tmp = img_detect[i * 48: (i + 1) * 48, j * 48: (j + 1) * 48, :]
                tmp = cv2.resize(tmp, (224, 224))
                start_time = time.time()
                x = image.img_to_array(tmp)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feature = model_mobilenet.predict(x).reshape((1, 7*7*1024))
                #print(model_smoke_detector.predict(feature)[0, 0])
                result = model_smoke_detector.predict(feature)[0, 0]
                detection_results.append(result)
                end_time = time.time()

                total_time += end_time - start_time
                count += 1'''
        
        img_detect = cv2.resize(img[:, :, [2, 1, 0]], (224, 224))
        #tmp = img_detect[i * 48: (i + 1) * 48, j * 48: (j + 1) * 48, :]
        #tmp = cv2.resize(tmp, (224, 224))
        start_time = time.time()
        x = image.img_to_array(img_detect)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model_mobilenet.predict(x).reshape((1, 7*7*1024))
        #print(model_smoke_detector.predict(feature)[0, 0])
        result = model_smoke_detector.predict(feature)[0, 0]
        #detection_results.append(result)
        end_time = time.time()

        total_time += end_time - start_time
        count += 1
                
        #frames.pop(0)
        #frames.append(1 if result > 0.5 else 0)
        #img = cv2.flip(img, 1)
        cv2.putText(img,
                    'Probability: {0}'.format(str(result)),
                    (30, 30), 
                    cv2.FONT_HERSHEY_COMPLEX, 
                    1, 
                    (0, 0, 255), 
                    1)
        cv2.putText(img,
                    '{0}'.format('smoke' if result > 0.5 else 'no smoke'),
                    (30, 90), 
                    cv2.FONT_HERSHEY_COMPLEX, 
                    2, 
                    (0, 0, 255), 
                    2)
        '''cv2.putText(img,
                    '{0}'.format('smoke' if sum(frames) > 0 else 'no smoke'),
                    (30, 90), 
                    cv2.FONT_HERSHEY_COMPLEX, 
                    2, 
                    (0, 0, 255), 
                    2)'''
        cv2.imshow('', img)
        key = cv2.waitKey(steps[choice])
        #video_writer.write(img)
        if key == 27:
            break
        elif key == ord('n'):
            choice += 1
            choice %= 3
    else:
        cv2.destroyAllWindows()
        break
cap.release()
#video_writer.release()


# In[5]:


total_time / count

