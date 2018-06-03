# Lip-reading-by-CNN-and-LSTM-architecture



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
  
![fig_2](./figures/fig_2.PNG)
 
  Artificial neural network is compounded of artificial neuron. Neural network is composed of layer and the layer has several neurons. There is input layer, output layer and hidden layer. Each neurons are fully-connected as above picture. If the number of hidden layer increase, the neural network is called as ‘Deep Neural Network (DNN)’. And, deep learning use this deep neural network for learning model. In hidden layer, generally use ReLU for activation function. The meaning of ‘Training’ or ‘Learning’ in the deep learning is adjustment of weights. The detailed explanation of ‘Learning’ exists below.
 
  There are many types of neural network or deep learning model. In this text, learn about CNN and RNN that are used in our model and famous models.
