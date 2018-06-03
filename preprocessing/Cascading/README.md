# Casacding Lip Tracker Algorithm

<br />
  
![fig_1](./figures/fig_1.png)<br />
**Cascading lip tracker algorithm**

<br />

## Load the raw data

### Import library - openCV, numpy, os

~~~python
import numpy as np
import cv2
import os
~~~

  This step contains importing library of python. Cv2, numpy and os means openCV, numpy, os library respectively. OpenCV (Open Source Computer Vision) is a library of programming functions mainly aimed at real-time computer vision. OpenCV’s application areas include: 2D and 3D feature toolkits, Facial recognition system, Motion tracking, etc. OpenCV is written in C++ and its primary interface is in C++. But there are bindings in Python, Java and so on. We will use the Python. Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. Os is a library for setting a path of the data. We will use os library to make a directory of the data.

### Load 'mouth casacade classifier' file

~~~python
mouth_cascade = cv2.CascadeClassifier('./node-opencv-master/node-opencv-master/data/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')
~~~

  Function of openCV, cv2.CascadeClassifier is used to load a classifier file.

  (https://docs.opencv.org/3.4.1/d1/de5/classcv_1_1CascadeClassifier.html) 

  The classifier file is trained with a few hundred sample views of a particular object (i.e., a face or a car), called positive examples, that are scaled to the same size (say 20*20), and negative example – arbitrary images of the same size. The file studies with some features of particular objects, such as edge feature, line features, and so on. The word “cascade” in the classifier name means that the resultant classifier consists of several simpler classifiers (stages) that are applied subsequently to a region of interest until at some stage the candidate is rejected or all the stages are passed
  
<br />
  
![fig_2](./figures/fig_2.png)<br />
**Features used by training**

<br />

  You can get additional information by site below.

  (https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html) 
  
  We used ‘Mouth cascade classifier’ file, “cascade haarcascade_mcs_mouth.xml”. This file is uploaded in Github. We just downloaded it, and used it. (https://github.com/peterbraden/node-opencv/tree/master/data) 

  If the file is empty, error would encounter you.

### Load raw data

~~~python
for i in range(1, 101):
    cap = cv2.VideoCapture('./deepset/news1/%d.avi' % i)
    # recommend avi file, mp4 file often cause some error for loading video
~~~

  We will load raw data from each label. Because the amount of raw data of each label is 100, ‘for’ statement has to be repeated 100 times.
  
  **Cv2.VideoCapture** is the function of openCV. It is used to load a video. You can load a video file by typing a name of video file. If you type 0 in the function (cv2.VideoCapture(0)) instead of the file name, you could load your own webcam. 
  
  (https://docs.opencv.org/3.2.0/d8/dfe/classcv_1_1VideoCapture.html) 
