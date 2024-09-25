---
layout: post
title:  "Generative Pre-training"
date:   2023-08-13 10:00:10 +0530
categories: AI
---

# Introduction 🚀

In this blog, we will explore the main idea behind chat-GPT, which is generative pre-training, and create our own smaller version of a question-answering model using generative pre-training. 🤖💬

The research paper we will be referencing is "Improving Language Understanding by Generative Pre-Training." 📚🔬

# Generative Pre-Trained Transformer (GPT) 🧠

#### Transfer Learning 🔄

Transfer learning is a machine learning technique in which a model trained on a large dataset is adapted for a different but related task. 🎓➡️🆕

The concept behind transfer learning is that the pre-trained model has learned general patterns and can be adjusted to specific domains or tasks with minimal additional training data. This saves time and resources compared to training a model from scratch. ⏳💰

#### Generative Pre-training 🔮

In the field of natural language processing, generative pre-training is an example of transfer learning. A language model is pre-trained on a large corpus of text data using self-supervised learning, and then fine-tuned on a smaller, labeled dataset for a specific task. 📚➡️🎯

The Generative Pre-trained Transformer (GPT) is a state-of-the-art deep learning architecture developed by OpenAI for natural language processing tasks, such as text generation, text completion, and text classification. 🌟🤖

GPT models are usually pre-trained on a large corpus of text data from the internet to learn general language patterns. Then, they are fine-tuned on specific domains or tasks using smaller, domain-specific datasets to adapt them to the specific context and language patterns of the target domain. 🌐➡️🎯

This is a semi-supervised approach that involves unsupervised pre-training and supervised fine-tuning. 🔄🏋️

During pre-training in generative pre-training, the model is trained to predict missing words or generate new text based on the input context. The goal is to learn a universal representation that transfers with little adaptation to a wide range of tasks. 🧩🔮

In the context of generative pre-training, "generative" refers to the ability of the model to generate new output based on the input it has learned. In other words, the model is capable of generating new text that is similar to the text it was trained on, but not identical. This is distinct from discriminative models, which are used for tasks such as classification and are designed to distinguish between different input classes. 🎨🆚🔍

#### Self-Supervised Learning 🤖🔄

Self-supervised learning is a type of machine learning in which the model is trained to predict or generate missing information in the data without explicitly being told what the missing information is. This is different from supervised learning, where the model is given labeled data to learn from. 🕵️‍♂️🧠

An example of self-supervised learning is image inpainting, where the model is trained to fill in missing pixels in an image based on the context of the surrounding pixels. In this case, the model is not explicitly told which pixels are missing, but rather it learns to infer the missing pixels based on the patterns it has learned in the input image. 🖼️🔍

Self-supervised learning is a powerful approach because it allows the model to learn from large amounts of unlabeled data, which is often easier to obtain than labeled data. This can be especially useful in domains where labeled data is scarce or expensive to obtain. 📊💡

# Training Framework 🏋️‍♂️

In the context of generative pre-training, the training framework typically involves two stages: pre-training and fine-tuning. 🔄

1. Pre-training 🎓
    
    During pre-training, the model is trained on a large, unlabeled corpus of text data using a self-supervised learning approach. This involves predicting missing words or generating new text based on the context of the input. 🔮
    
    The goal of pre-training is to learn a general representation of language that can be fine-tuned for specific tasks. 🌐
    
2. Fine-tuning 🎯
    
    During fine-tuning, the pre-trained model is further trained on a smaller, labeled dataset specific to the task at hand. This involves adjusting the parameters of the pre-trained model to better fit the specific context and language patterns of the target domain. 🔧
    
    Fine-tuning allows the model to adapt to the specific task and improve its performance. 📈
    

Overall, the goal of the training framework in generative pre-training is to learn a general representation of language that can be fine-tuned for a wide range of tasks with minimal additional training data. 🚀

## Unsupervised Pre-Training 🤖🔮

To perform generative pre-training, an unsupervised corpus of tokens $\mu = (u_1, …, u_n)$ is required. The language modeling objective in this context refers to predicting the next word in a sequence given the previous words. The goal is to maximize the probability of the next word in the sequence given the preceding words.

To achieve this, we use the standard language modeling objective and maximize:

$L_1(U) = \sum_i log P(u_i \lvert u_{i−k}, . . . , u_{i−1}; Θ)$

where k is the size of the context window. The conditional probability is modeled using a neural network with parameters $Θ$, which are trained using stochastic gradient descent. 

We use a multilayer transformer decoder for the language model, this is used to calculate the probability of next token given a sequence of tokens.

$h_0 = UW_e +W_p$, calculate the initial hidden state of the model.

where:

$U = (u_{-k}, . . . , u_{−1})$ is the context vector of tokens.

$W_e$ is the token embedding matrix, which maps each token to a high-dimensional vector representation.

$W_p$ is the position embedding matrix, which encodes the relative position of each token in the input sequence.

This hidden state is then processed through multiple layers of transformer blocks to produce the final output of the model. Each layer producing a new block of data $h_l$ that serves as the input to the next layer.

$h_l = transformer\_block(h_{l−1})∀i ∈ [1, n]$

where:

n is the number of layers in the transformer block.

We then calculate the probability distribution of the next word in a sequence given the previous words. The softmax function is applied to the dot product of the hidden state vector ($h_n$) and the transpose of the word embedding matrix ($W_e$) to obtain the probabilities of the next word.

$P(u) = softmax(h_nW^T_e )$

## Supervised Fine-Tuning 🎯🏋️‍♂️

We use a labeled dataset C and a sequence of input tokens $x^1, x^2, …., x^m$ and a label y. These inputs are passed through the pre-trained model to obtain the final output $h_l^m$, which is then fed to a linear output layer with parameter $W_y$.

This equation $P(y\lvert x^1, . . . , x^m) = softmax(h_l^m W_y)$ calculates the probability of target output y given a set of input tokens $x^1, . . . , x^m$. 

where:

$h_l^m$ represents the hidden state of the model after processing the input sequence (of size m) through l layers of transformer blocks.

$W_y$ is the parameter matrix of the fine-tuned model.

The final objective is to maximize:

$L_2(C) = \sum_{(x, y)}log P(y\lvert x^1,...,x^m)$ i.e. the objective is to maximize a function $L_2(C)$, which is a sum of logarithmic probabilities. 

Including language modeling as an auxiliary objective to fine-tuning helps learning by improving the generalization of the supervised model and accelerating convergence. To achieve this, we add $L_1(C)$ language modeling objective with weight $\lambda$. Therefore, the final objective to maximize is:

$L_3(C) = L_2(C) + \lambda * L_1(C)$

# Task-specific Input Transformations 🔄

We use a traversal-style approach to convert structured inputs into an ordered sequence that our pre-trained model can process.

![Fig: **(Left) Transformer architecture and training objectives used in this work.
      (Right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear + softmax layer.**](/assets/2024/September/gpt.png)

*Fig: **(Left) Transformer architecture and training objectives used in this work.
      (Right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear + softmax layer.***

1. Textual Entailment 🧩
    
    For entailment tasks, we concatenate the token sequences of the premise p and hypothesis h, with a delimiter token ($) in between.
    
2. Similarity 🪞
    
    For similarity tasks, the input sequence contains both possible sentence orderings (with a delimiter in between). We process each independently to produce two sequence representations $h_l^m$.
    
3. Question Answering and Commonsense Reasoning 💡
    
    For these tasks, we are given a context document z, a question q, and a set of possible answers $\{a_k\}$. We create a sequence of [z;q;$;a_k].
    

Each of these sequences is processed independently with our model and then normalized via a softmax layer to produce an output distribution over possible answers.

### Implementation

You can find the Implementation of fine Tuning this GPT2 model on question-answering dataset here .

[Kaggle](https://www.kaggle.com/code/dsmeena/pytorch-fine-tuning-gpt2)

### References

[Papers with Code - GPT Explained](https://paperswithcode.com/method/gpt)