# Lip-reading-by-CNN-and-LSTM-architecture

<br />
<br />

## Introduction
  This deep learning project is about Lip Reading which is a technique of understanding speech by visually interpreting the movements of the lips so, we implemented this Lip Reading by using deep learning. It can be used for hard-of hearing people or to get some information from video without sound.
  
### Objective
  Lip Reading relies on the kind of the language and, in this project, **we chose Hangul as the language to implement the Lip Reading**.

  Because there were no Hangul lip datasets available for deep learning, it was necessary to create the datasets manually. Therefore, to recognize the overall general terms of language, not only does it require a large number of datasets, but also the size and complexity of the neural net increases. Due to time and hardware limitations, **this project defined the problem by classifying only a few words**.
  
### Precedent research & Reference
  There are some precedent researches which are related to this project as follows :
  
  * Garg Amit, Jonathan Noyola, and Sameep Bagadia. **Lip reading using CNN and LSTM**. Technical report, Stanford University, CS231n project report, 2016.
  
  * Gutierrez, Abiel, and Zoe-Alanah Robert. "**Lip Reading Word Classification**".
  
  * Parth Khetarpal, Shayan Sadar, and Riaz Moradian. "**LipVision: A Deep Learning Approach**“, International Journal of Computer Applications, 2017.



## Background Knowledge for Deep Learning

### What is the deep learning?

  Deep learning is defined by class of machine learning algorithms that use diverse combination of nonlinear processing for abstract. The meaning of abstract is feature extraction and transformation. In other words, **teach computer to human’s way of thinking**.
  
  Deep learning is learned by using **artificial neural network**. So, we learn about artificial neural network in the next paragraph.

### Artificial Neural Network

  Artificial neural network is imitation of biological neural network as the word itself. First we learn about an artificial neuron. The below left side picture is representation biological neuron. The neuron is an electrically excitable cell, receive electrical signals from other neurons and then give this signal to other neurons.
  
![fig_1](./figures/fig_1.PNG)
 
  An artificial neuron is a mathematical function expressed as a concept of biological neuron. The artificial neuron receives one or more inputs(x). There exist individual weights(w) for inputs. If receive 3 inputs, then there exist 3 weights. The summation results is total sum of input(x) multiplied by weight (w) and bias input (b). This summation is passed through a non-linear function known as an activation function (f). The result value of activation function is an output of artificial neuron. This process is expressed in an above right side picture. There is many activation functions such as sigmoid, ReLU, softmax. The output can change by activation function even if same summation.
  
![fig_2](./figures/fig_2.png)
 
  Artificial neural network is compounded of artificial neuron. Neural network is composed of layer and the layer has several neurons. There is input layer, output layer and hidden layer. Each neurons are fully-connected as above picture. If the number of hidden layer increase, the neural network is called as ‘Deep Neural Network (DNN)’. And, deep learning use this deep neural network for learning model. In hidden layer, generally use ReLU for activation function. The meaning of ‘Training’ or ‘Learning’ in the deep learning is adjustment of weights. The detailed explanation of ‘Learning’ exists below.
 
  There are many types of neural network or deep learning model. In this text, learn about CNN and RNN that are used in our model and famous models.
  
### Convolution Neural Network

  Convolutional neural network is a class of deep, feed-forward neural networks, generally used to analyzing visual imagery. CNN is composed of convolution layer and pooling layer, flatten layer, fully connected layer. CNN can be separated region that feature extraction of picture and region that classification. The convolution layer and pooling layer are region that feature extraction. And, the fully connected layer is region that classification. The flatten layer is located in between this regions.
  
![fig_3](./figures/fig_3.png)
</br>
**The process of convolution product by 2X2 filter with 1 stride**


  The image is composed of pixels. Pixel is a real number. And, each pixel express RGB 3 real numbers is color image. For example, the data shape of 24X24 pixel black and white image is (24,24,1). If the input data is image, the number of weights increase prodigiously. By this problem, calculate summation and activation function about only adjacent region. This is concept of filter. The number of filter’s weights is same as filter’s size. For example, 3X3 filter has 9 weights. The meaning of ‘Learning’ in the convolutional neural network is adjustment of filter’s weights. The filter move on input data(image) and perform convolution product. The result value is located in present filter’s location. The distance interval of filter’s moving is stride. The matrix that composed of result values is called ‘Feature Map’. The ‘Activation Map’ is a result of applied activation function on feature map. So, the output of convolution layer is activation map. There can exist several filters. The output of convolution layer is smaller than input data.

  (Padding is a method for avoid reduction of data’s size by convolution layer. Stuff a shell of data with specific value generally 0.)

  Pooling layer accept the output of convolution layer as input data. By pooling layer, the data size is reduce and some feature are extracted. One of the pooling is Max pooling. The max pooling extract maximum value in the filter size. By max pooling, can reduce noise and minor elements. The below picture represents max pooling layer.

![fig_4](./figures/fig_4.png)

  Flatten layer transform 2D data such as matrix to 1D data. The data must 1D shape for input data of fully-connected layer. So, flatten layer is located in between convolution layer or pooling layer and fully-connected layer.
 
![fig_5](./figures/fig_5.png)

  This above picture is basic structure of CNN. Convolution layer and pooling layer are stacked repeatedly. And the output transform 1D data by flatten layer, the transformed 1D data input fully-connected layer. The output layer apply Softmax activation function for classification.
 
  In the “Basic Keras Programming tutorial code”, you can study how to construct the CNN model in the Keras.
  
### Recurrent Neural Network

  Recurrent neural network is typically used to find regular pattern. Sequence is the data that important order. RNN’s input is a sequence. Recurrent neural network consider present input data and previous input data.
