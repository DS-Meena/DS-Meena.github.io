---
layout: post
title:  "Demystifying Neural Networks"
date:   2023-06-11 15:08:10 +0530
categories: AI
---

# Introduction

In this blog, we will explore the fascinating world of neural networks. We will start with the basics of neural networks and their work. Then we will dive into the different types of neural networks and their applications, including deep learning and convolutional neural networks. Finally, we will discuss some of the challenges and limitations of neural networks and their potential for future advancements. Whether you are a beginner or an expert, this blog will demystify neural networks and provide a deeper understanding of this exciting field.

## Neural Networks in our Brain?
Neural networks in our brain are complex network of neurons that communicate with one another through electrical and chemical signals to process and transmit information throughout the body. They are the basis for many cognitive processes, including perception, learning, and memory.


# Artificial Neural Network

These are mathematical models for learning that are inspired by biological neural networks.
The models use mathematical functions to map inputs to outputs based on the network’s structure and parameters.
They also allow for the learning of the network’s parameters based on data.

**Hypothesis** is a function that maps inputs to outputs in a neural network. The hypothesis function is defined by the network’s structure and parameters, and it represents the network’s prediction for a given input.

**Activation functions** are mathematical functions that are applied to each neuron’s output in a neural network. They introduce non-linearity into the network, essential for learning complex functions. Common activation functions include sigmoid, ReLU, and tanh.

**Example:**

The hypothesis function for a simple neural network with one input and one output might be:

$h_\theta(x) = {\sigma} {(\theta_0 + \theta_1 x)}$

where θ0 and θ1 are the parameters of the network and σ is the activation function.

## Types of Activation Function

There are several types of activation functions, each with unique properties that make them suitable for different types of problems. Some of the most common activation functions are:

#### 1. Step function

The step function is a simple activation function that outputs a 1 if the input is greater than or equal to 0, and a 0 otherwise. The step function is rarely used in practice because it is not differentiable, which makes it difficult to use with gradient-based optimization algorithms.

$$f(x) = \begin{cases} 1, & \text{if } x \geq 0 \\\ 0, & \text{if } x < 0 \end{cases}$$

#### 2. Sigmoid function

The sigmoid function is a common activation function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where z is the input to the neuron. The output of the sigmoid function is always between 0 and 1, which is useful for representing probabilities in classification tasks.

#### 3. ReLU (Rectified Linear Unit)

The ReLU (Rectified Linear Unit) function is a commonly used activation function in neural networks. It is defined as:

$$f(x) = \begin{cases} x, & \text{if } x > 0 \\ 0, & \text{otherwise} \end{cases}$$

The ReLU function is computationally efficient and has been shown to work well in practice, particularly for deep neural networks. It is also easy to implement and does not suffer from the vanishing gradient problem that can occur with other activation functions such as the sigmoid function.

#### 4. Tanh (Hyperbolic Tangent)

Tanh (Hyperbolic Tangent) is a commonly used activation function in neural networks. It is defined as: 

$$\tanh(z) = \frac{e^z – e^{-z}}{e^z + e^{-z}}$$

The tanh function is similar to the sigmoid function, but its output ranges from -1 to 1, which can be useful for certain types of problems. It is also differentiable, making it suitable for gradient-based optimization algorithms.

#### 5. SoftMax

The SoftMax function is often used as an activation function in the output layer of neural networks for multi-class classification problems. It takes a vector of real-valued inputs and outputs a probability distribution over the classes.

The softmax function is defined as:

$$softmax(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

where zj is the input to the j-th output neuron, and K is the total number of output neurons. The softmax function ensures that the outputs of the network form a valid probability distribution, with each output representing the probability of the input belonging to a particular class.

![Fig: Activation functions and their derivatives (O’Reilly)](/assets/2024/September/activation-functions.png)

Activation functions are non-linear, allowing our neural network to solve complex problems (like non-linear separable classes).

### How to optimize weights

We have a separate post to understand the underlying algorithm to train ANN. Check the post below to learn about the gradient descent algorithm.Gradient Descent

[Gradient Descent](https://ds-meena.github.io/ai/2024/08/03/Gradient_descent.html)

# Perceptron

**Perceptron** is a type of artificial neural network that consists of a single layer of binary threshold neurons. It can be used for binary classification problems and is trained using the perceptron learning algorithm, which updates the weights of the network based on the error between the predicted output and the true output. The perceptron is a simple model, but it can be effective for linearly separable problems.

![Fig: Perceptron (Source: Deepai)](/assets/2024/September/perceptron-6168423.png)

The output of a perceptron, $\vec y = \text{step function}( \vec w . \vec x + b)$

where:

$\vec {w}$ = connection weights of synapses between input neurons and our only layer

b = bias term added to the weighted sum of the inputs in the neuron

Note that a constant 1 is often used as the bias term in a perceptron. The bias term has an associated weight $w_0$; hence, $w_0 \cdot 1$ is added to the weighted sum of the inputs.


### Perceptron learning algorithm (weight update):

Till the algorithm does not converge, keep updating the weights and biases using the below equations (in accordance with the gradient of MSE w.r.t to weights and bias).

weight update: $\vec w^\text{(next step)} = \vec w + \eta(y - \hat y)\vec x$

$b^\text{(next step)} = b + \eta(y - \hat y)$

where,

$\hat y$ is our predicted output

$\eta$ is our learning rate

The perceptron learning rule updates weights incrementally based on the classification error, adjusting weights after each misclassified example.

The Perceptron learning algorithm guarantees convergence to a solution with zero error if the training data is linearly separable. If the data is not linearly separable, the algorithm may not converge and will not achieve zero error.

### Perceptron limitations

The perceptron has several disadvantages, including:
- It can only solve linearly separable problems, which limits its applicability to more complex problems.
e.g. why perceptron for xor is not possible.

![alt text](/assets/2024/September/xor-1.png)

- It may converge to a suboptimal solution if the training data is not linearly separable or if the learning rate is too high.
- It requires labelled training data, which can be expensive or time-consuming to obtain.
- It can suffer from the problem of overfitting, where the model performs well on the training data but poorly on new, unseen data.


# Multilayer Perceptron (MLP)

A multilayer neural network (MNN) or Multilayer perceptron (MLP) is an artificial neural network with an input layer, an output layer, and at least one hidden layer. The hidden layers are composed of neurons that transform the input data into a form that can be used by the output layer. The output layer produces the final output of the network, which is usually a prediction or classification.

MLPs can learn complex nonlinear relationships between inputs and outputs, making them suitable for a wide range of applications, including image recognition, natural language processing, and speech recognition.

![alt text](/assets/2024/September/1__M4bZyuwaGby6KMiYVYXvg.jpg)

*Fig: MNN (Source: towardsdatascience.com)*


However, if it's too complex, it may overfit to noise in the data you use to train it and be a poor predictor of new data.

## How to optimize weights of hidden layers
You can learn more about how backpropagation efficiently calculates the gradient w.r.t network parameters for all layers and updates the weights of hidden layers.

[Gradient Descent](https://ds-meena.github.io/ai/2024/08/03/Gradient_descent.html)

# Deep Neural Network

A Deep Neural Network (DNN) is an artificial neural network with multiple hidden layers between the input and output layers. DNNs are capable of learning complex nonlinear relationships between inputs and outputs, making them suitable for a wide range of applications, including image recognition, natural language processing, and speech recognition.

DNNs are typically trained using backpropagation. The choice of activation function and optimization algorithm can have a significant impact on the performance and convergence rate of a DNN. Common activation functions include ReLU, sigmoid, and tanh, and common optimization algorithms include stochastic gradient descent, mini-batch gradient descent, and Adam.

DNNs have achieved state-of-the-art performance on a wide range of tasks, including image classification, speech recognition, and natural language processing.

![alt text](/assets/2024/September/Two-or-more-hidden-layers-comprise-a-Deep-Neural-Network.png)

*Fig: Deep Neural network (Source: bmc.com)*

## Overfitting

One of the key challenges in training DNNs is the problem of overfitting, where the model performs well on the training data but poorly on new, unseen data. It become over reliant on some neurons.

This problem can be addressed through techniques such as regularization, early stopping, and data augmentation.

### 1. Regularization

**Regularization** is a technique used in machine learning to prevent overfitting and improve the generalization of the model.

It works by adding a penalty term to the cost function that discourages the model from fitting the training data too closely.

Penalty term could be L1 norm, L2 norm or probability to drop.
There are several types of regularization techniques, including:

#### L1 and L2 Regularization

L1 and L2 regularization are two commonly used regularization techniques in machine learning. They work by adding a penalty term to the cost function that is proportional to the norm of the model’s parameters, and it penalizes the model for having large parameter values.

The L1 norm of a vector is defined as the sum of the absolute values of its components:

$\|x\|1 = \sum{i=1}^n \lvert x_i\lvert$

The L2 norm of a vector is defined as the square root of the sum of the squares of its components:

$\|x\|2 = \sqrt{\sum{i=1}^n x_i^2}$

The L1 norm tends to produce sparse parameter vectors, where many of the components are zero. This can be useful for feature selection and reducing the complexity of the model.

The L2 norm tends to produce parameter vectors with small, non-zero components. This can be useful for preventing overfitting and improving the generalization of the model.

#### Dropout

Dropout is a regularization technique that works by randomly dropping out (setting to zero) a fraction of the neurons in the network during training. This forces the network to learn more robust features that are not dependent on the presence of specific neurons.

Dropout has been shown to be effective at reducing overfitting in a wide range of neural network architectures and applications.

Example →

During training, each neuron has a probability

pof being kept, and a probability1-pof being dropped out. The value ofpis typically set to 0.5, but can be tuned as a hyperparameter.

When the network is evaluated on new data, all of the neurons are used, but their output is scaled by a factor ofpto account for the dropout during training.

Regularization can be applied to a wide range of machine learning models, including linear regression, logistic regression, and neural networks.

It is a powerful technique for improving the performance and generalization of machine learning models, especially when the amount of available training data is limited.

### 2. Early Stopping

Early stopping is a technique used in machine learning to prevent overfitting and improve the generalization of the model. It works by monitoring the performance of the model on a validation set during training and stopping the training process when the performance on the validation set stops improving.

To implement early stopping, we need to split the dataset into training, validation, and test sets. We use the training set to train the model, the validation set to monitor the performance of the model during training, and the test set to evaluate the performance of the final model.

The number of epochs to wait before stopping the training process is a hyperparameter that needs to be tuned. If we stop the training process too early, we may not have trained the model enough to achieve its full potential. If we wait too long, we may overfit the training data and generalize poorly to new, unseen data.

Early stopping is a powerful technique for improving the generalization performance of machine learning models, especially when the amount of available training data is limited.

### 3. Data Augmentation

**Data Augmentation** is a technique used in machine learning to increase the size of the training set by generating new data from the existing data. This can be useful for improving the performance and generalization of the model, especially when the amount of available training data is limited.

There are several types of data augmentation techniques, including:

#### 1. Image Augmentation

Image augmentation is a common technique used in computer vision to generate new images from the existing images.
Some of the common image augmentation techniques include:

- Flipping the image horizontally or vertically
- Rotating the image by a certain angle
- Scaling the image up or down
- Adding noise to the image
- Changing the brightness or contrast of the image

These techniques can be used to generate new images that are similar to the existing images, but with slight variations that can help the model learn to be more robust to different types of input.

#### 2. Text Augmentation

Text augmentation is a technique used in natural language processing to generate new text from the existing text.

Some of the common text augmentation techniques include:

- Synonym replacement: replacing words with their synonyms
- Random insertion: inserting new words at random positions in the text
- Random deletion: deleting words at random positions in the text
- Random swap: swapping two words at random positions in the text.

These techniques can be used to generate new text that is similar to the existing text, but with slight variations that can help the model learn to be more robust to different types of input.

#### 3. Audio Augmentation

Audio augmentation is a technique used in speech recognition to generate new audio data from the existing audio data.

Some of the common audio augmentation techniques include:

- Adding noise to the audio
- Changing the speed or pitch of the audio
- Combining multiple audio files into a single file
- Cutting and splicing the audio at random positions.
These techniques can be used to generate new audio data that is similar to the existing audio data, but with slight variations that can help the model learn to be more robust to different types of input.

![alt text](/assets/2024/September/augmentation.jpg)

*Fig: Image augmentation (Source: paperswithcode.com)*


Data augmentation can be a powerful technique for improving the performance and generalization of machine learning models, especially when the amount of available training data is limited.

# Computer Vision

Computer Vision is a field of study focused on enabling computers to interpret and understand visual information from the world around us. It involves the development of algorithms and techniques for processing, analyzing, and interpreting images and video data.

Computer vision is used in a wide range of applications, including:

- Object detection and recognition
- Image and video classification
- Face and gesture recognition
- Autonomous vehicles and robotics
- Medical imaging
- Augmented and virtual reality

Some of the key techniques used in computer vision include:

- **Image filtering** – This involves applying a filter or mask to an image to extract or enhance certain features, such as edges or textures.
- **Feature detection** – This involves identifying key features in an image, such as corners or blobs, that can be used for object recognition or tracking.
- **Object recognition** – This involves identifying objects in an image or - video sequence and classifying them into different categories.
- **Image segmentation** – This involves dividing an image into different regions or segments based on their visual properties, such as color or texture.
- **Deep learning** – This involves training artificial neural networks to learn and recognize patterns in image or video data.

Computer vision is an important area of research and development, with many applications in industry, healthcare, and entertainment.


## Convolution Neural Network (CNN)

A **Convolutional Neural Network (CNN)** is a type of neural network that is particularly suited for image recognition and other computer vision tasks. It is composed of a series of convolutional layers that extract features from the input image, followed by one or more fully connected layers that perform the classification or regression task.

Convolutional layers apply a set of filters (also called kernels or weights) to the input image, producing a set of feature maps that highlight different aspects of the image, such as edges or textures. The filters are learned during training, and can be thought of as feature detectors that learn to recognize specific patterns or structures in the image.

Pooling layers are often used after the convolutional layers to reduce the spatial dimensions of the feature maps, which helps to reduce the computational requirements of subsequent layers in the network and prevent overfitting.

The fully connected layers take the output of the convolutional layers and produce the final output of the network, which is usually a prediction or classification.

![alt text](/assets/2024/September/Schematic-diagram-of-a-basic-convolutional-neural-network-CNN-architecture-26.png)

**Fig: CNN (Source: Upgrad.com)**

CNNs have achieved state-of-the-art performance on a wide range of image recognition tasks, including object detection, segmentation, and classification. They are also used in other computer vision tasks, such as video recognition and natural language processing.
Training a CNN can be computationally expensive, especially for large datasets and complex architectures, but there are many pre-trained models available that can be fine-tuned for specific tasks.

### Image Convolution

Image convolution is a image filtering technique. It involves applying a kernel or filter to an image to compute a new pixel value at each location in the image. The kernel is a small matrix of weights that is convolved with the image to produce a new image with modified pixel values.

The most commonly used kernel in image convolution is the **Gaussian kernel**, which is used to smooth images and reduce noise. Other kernels are used for different purposes, such as edge detection or feature extraction.

The convolution operation can be expressed mathematically as follows:

$(f*g)(x,y) = \sum_{i=-a}^{a} \sum_{j=-b}^{b} f(i,j) g(x-i,y-j)$

where

f is the input image,g is the kernel, and a and b are the half-widths of the kernel.

Image convolution is a powerful technique for extracting features from images, and it is used in many computer vision and image processing applications, such as object detection, recognition, and tracking.

![Convolution Operation](/assets/2024/September/Convolution.com\)
Fig: Applying convolution kernel (Source: Nvidia.com)


## Pooling

Pooling is a technique used in computer vision and image processing to downsample an image or feature map by applying a function, such as max or average pooling, to non-overlapping blocks of the input.

Pooling helps to reduce the spatial dimensions of the input, which can reduce the computational requirements of subsequent layers in the network and help to prevent overfitting.

Pooling is commonly used in convolutional neural networks (CNNs) for image classification and other computer vision tasks.

**Max pooling** is a pooling operation that outputs the maximum value within each non-overlapping block of the input.

Max pooling helps to reduce the spatial dimensions of the input and can help to prevent overfitting by reducing the number of parameters in the model.

![alt text](/assets/2024/September/MaxpoolSample2.png)

*Fig: Max pooling (Source: Papers with code)*


**Average pooling** is a pooling operation that outputs the average value within each non-overlapping block of the input.

Average pooling helps to reduce the spatial dimensions of the input and can help to prevent overfitting by reducing the number of parameters in the model.

![alt text](/assets/2024/September/Screen_Shot_2020-05-24_at_1.51.40_PM.png)

*Fig: Average Pooling (Source: Papers with code)*


# Feed-Forward Neural Network

A feed-forward neural network is a type of artificial neural network in which the information flows in one direction, from the input layer to the output layer, with no loops or cycles in the network. The input layer receives the input data, which is then transformed by the hidden layers using a set of weights and biases. The output layer produces the final output of the network, which is usually a prediction or classification.

Feed-forward neural networks are typically trained using backpropagation, which involves computing the gradient of the cost function with respect to the network’s parameters and using this gradient to update the weights and biases.

Feed-forward neural networks are capable of learning complex nonlinear relationships between inputs and outputs, making them suitable for a wide range of applications, including image recognition, natural language processing, and speech recognition.

However, feed-forward neural networks **have several limitations**, including:

- They are not well-suited for tasks that require processing of sequential or time-series data, such as language translation or speech recognition.
- They can suffer from the problem of vanishing gradients (because use backpropagation), where the gradient becomes very small as it is propagated backwards through the network, making it difficult to update the weights and biases of the early layers in the network.
- They are not well-suited for tasks that require memory or attention, such as question answering or dialogue generation.

Despite these limitations, feed-forward neural networks have been successfully applied to a wide range of tasks and continue to be an active area of research in the field of artificial intelligence.

# Recurrent neural Network

A **Recurrent Neural Network (RNN)** is a type of neural network that is particularly suited for processing sequential data, such as time series or natural language.
It is composed of a series of **recurrent layers**, these layers process the input sequence one element at a time, while maintaining a hidden state that summarizes the information learned from previous elements in the sequence. This hidden state is updated with each new input, allowing the network to capture the temporal dependencies in the input data.

![alt text](/assets/2024/September/17464JywniHv.png)

*Fig: Architecture of RNN (source: medium.com)*

We call Recurrent Neural Networks (RNNs) “recurrent” because they have a **hidden state** that is passed from one time step to the next, allowing the network to capture temporal dependencies in the input data. This recurrent structure is what distinguishes RNNs from other types of neural networks, such as feedforward networks, which do not have any memory of previous inputs or outputs.

- **Hidden State**
    
    The internal state of an RNN can be thought of as a “hidden” or “memory” vector that encodes information about the previous inputs to the network. This hidden state is updated at each time step by combining the current input (t) with the previous hidden state (t-1) using a set of learned weights and biases.

RNNs can be trained using backpropagation through time (BPTT), which involves computing the gradient of the cost function with respect to the network’s parameters over a sequence of inputs and using this gradient to update the weights and biases.

RNNs are capable of learning complex nonlinear relationships between sequential inputs and outputs, making them suitable for a wide range of applications, including language modeling, speech recognition, and machine translation.

- **Vanishing or exploding gradients**

    RNNs can suffer from the problem of vanishing or exploding gradients, where the gradient becomes very small or very large as it is propagated backwards through the network, making it difficult to train the network effectively over long sequences.

    This problem has been addressed through the development of more advanced RNN architectures, such as long short-term memory (LSTM) and gated recurrent unit (GRU) networks, which are designed to better handle long-term dependencies in the input data.

    LSTMs and GRUs are types of RNNs that are specifically designed to address the vanishing gradient problem. They use gating mechanisms to selectively update and forget information in the hidden state, which allows them to capture long-term dependencies and temporal correlations more effectively than traditional RNNs.

Overall, RNNs are a powerful tool for processing sequential data, and have been used to achieve state-of-the-art performance on a wide range of tasks in natural language processing, speech recognition, and other domains.

Recurrent Neural Networks (RNNs) are a powerful tool for processing sequential data, and have been used to achieve state-of-the-art performance on a wide range of tasks in natural language processing, speech recognition, and other domains. Here are some examples of applications that use RNNs:

1. **Language modeling**: RNNs are used to model the probability distribution over sequences of words in natural language. This is useful for tasks such as speech recognition, machine translation, and text generation.
2. **Speech recognition**: RNNs are used to recognize speech signals and convert them to text. They are particularly effective at modeling the temporal dependencies in speech signals, which can be used to improve the accuracy of speech recognition systems.
3. **Machine translation**: RNNs are used to translate text from one language to another. They are particularly effective at modeling the complex relationships between words in different languages, which can be used to improve the accuracy of machine translation systems.
Example → Google translation
4. **Image captioning**: RNNs are used to generate captions for images. They are particularly effective at modeling the semantic relationships between objects in images and the natural language used to describe them.
Example → Microsoft caption Bot
5. **Video analysis**: RNNs are used to analyze video data, including action recognition, video captioning, and video summarization. They are particularly effective at modeling the temporal dependencies in video data, which can be used to improve the accuracy of video analysis systems.
Example → Youtube.com
6. **Handwriting recognition**: RNNs are used to recognize handwritten text and convert it to digital text. They are particularly effective at modeling the temporal dependencies in handwritten text, which can be used to improve the accuracy of handwriting recognition systems.

Overall, RNNs are a versatile tool for processing sequential data, and have been used to achieve state-of-the-art performance on a wide range of tasks in natural language processing, speech recognition, and other domains.

That’s all for this blog, hope this will teach you something useful 💖💖.