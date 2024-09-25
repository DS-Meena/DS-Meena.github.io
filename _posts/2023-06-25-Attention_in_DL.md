---
layout: post
title:  "Attention in DL"
date:   2023-06-25 10:00:10 +0530
categories: AI
---

Welcome to this blog post on attention in deep learning! In this post, we will explore the concept of attention and its importance in the field of deep learning. We will start by explaining what attention is and how it works, and then move on to different types of attention mechanisms and their applications. Whether you are new to deep learning or an experienced practitioner 🧑‍⚕️, this post is sure to provide valuable insights into one of the most important concepts in modern machine learning.

# Attention

Attention is a mechanism in neural networks that allows the model to focus 🔍 on specific parts of the input sequence when making predictions. It has been a breakthrough in natural language processing and computer vision.  

Attention was first introduced in 2014 by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, and since then, it has become a fundamental concept in deep learning.

![Fig: The attention mechanism improves model’s prediction. Taken from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](/assets/2024/September/attention%20mechanism.png)

*Fig: The attention mechanism improves model’s prediction. Taken from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*

## Working of Attention 👷

The attention mechanism assigns weights to each element in the input sequence, indicating their relevance to the current output. Here's how it works:

1. The attention weights 🏋️‍♀️ are computed based on the current state of the model and the entire input sequence.
2. The input sequence is multiplied ❌ element-wise by the attention weights, producing a sequence of weighted vectors. These vectors are then summed ➕ to obtain a single vector or context vector that captures the most relevant parts of the input sequence.
3. This weighted sum of the input sequence or context vector is then used to compute the next state of the model.
    
    How? by concatenating 🔗 the context vector with the decoder output and passing through a feedforward neural network to get the final output sequence which is again used to update decoder hidden state.
    

By focusing on specific parts of the input sequence that are most relevant to the current output, the attention mechanism improves the model's performance.

```
  Input Sequence  --> Encoder --> Attention --> Decoder --> Output Sequence
```

In this diagram, the input sequence is first passed through an encoder, which produces a set of encoded representations. The attention mechanism then computes weights for each encoded representation, indicating its relevance 🤔 to the current output. The weighted sum of the encoded representations is then passed through a decoder, which produces the final output sequence.

## Coding View 🧑‍💻

To write an attention model from scratch, you would need to define the input and output shapes of the model, as well as the layers to be used. The model would then need to be trained using a suitable loss function and optimizer.

Here is an example of how to implement an attention model using TensorFlow 🌊:

```python
from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, dot, Activation, concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Define the input sequence shape and size
sequence_length = 10
input_size = 32

# Define the encoder and decoder layers using LSTM cells
encoder_inputs = Input(shape=(sequence_length, input_size))
encoder_lstm = LSTM(64, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(sequence_length, input_size))
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Compute attention weights
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder_outputs], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_outputs])

# Define the output layer
output_layer = Dense(input_size, activation='softmax')
output = output_layer(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], output)

```

In this example, we first define the shape and size of the input sequence. The input vectors have a dimensionality of 32 and input sequence has a length of 10.

Let’s take a quick glance, at these terms. 🧐

| Term | Definition |
| --- | --- |
| Input Vectors | Represent the elements of the input sequence |
| Input Sequence | Input of the model |
| Input size | Refers to the dimensionality of the input vectors |
| Input sequence length | Refers to the number of elements in the input sequence. |

We then define the encoder and decoder layers 🧱 using LSTM cells, similar to the previous example. 

We then compute the attention weights by taking the dot product of the decoder outputs and encoder outputs, and passing the result through a softmax activation function 🥎 to obtain the attention weights. 

We then compute the context vector 📃 by taking the dot product of the attention weights and the encoder outputs. 

Finally, we concatenate 🔗 the context vector and the decoder outputs, and pass the result through the output layer to obtain the final output.

There are many variations and improvements that can be made depending on the specific task and dataset being used.

## Mathematical View 👩‍🔬

Here are the mathematical equations for the attention mechanism:

Let $h_t$ be the hidden state of the decoder at time $t$, and let $e_{i,j}$ be a score that measures the relevance of the ith encoder output ($h_i$) and the jth decoder hidden state ($h_j$).

The attention scores $e_{i,j}$ can be computed using various methods, such as dot product, additive, and multiplicative attention. 

In the **dot product method 🔵**, the scores are computed as the dot product of the encoder outputs and the decoder hidden state:

$e_{i,j} = h_i^T h_j$

In the **additive method ➕**, the scores are computed as the sum of two feedforward neural networks:

$e_{i,j} = v_a^T \tanh(W_a h_i + U_a h_j)$

where $v_a$, $W_a$, and $U_a$ are learnable weight matrices.

In the **multiplicative method ✖️**, the scores are computed as the dot product of the decoder hidden state and a learnable weight matrix, which is then multiplied element-wise with the encoder outputs:

$e_{i,j} = h_i^T W_a h_j$

where $W_a$ is a learnable weight matrix.

In all cases, the attention mechanism allows the model to focus 🔍 on specific parts of the input sequence that are most relevant to the current output, improving the performance of the model.

The attention weights $\alpha_{i,j}$ are computed using the softmax function 🥎:

$\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k=1}^{T_x} \exp(e_{i,k})}$

where $T_x$ is the length of the input sequence, 

The exponential function: $\exp(x) = e^x$

The context vector $c_t$ is then computed as a weighted sum ➕ of the encoder outputs, using the attention weights:

$c_t = \sum_{i=1}^{T_x} \alpha_{i,t} h_i$

Finally, the context vector is concatenated 🔗 with the decoder hidden state $h_t$ and passed through a feedforward neural network to obtain the final output.

This output represents the model's prediction or the next state of the model.

## Self-Attention 🤳

Self-attention, also known as intra-attention, is a type of attention mechanism where the input sequence is compared 🪞 to itself to obtain a set of attention weights. This allows the model to attend to different parts of the input sequence when making predictions, without requiring any additional context. 

### Working of self-attention 👷

In self-attention, the input sequence is first passed through three linear transformations to obtain query, key 🗝️, and value vectors. 

Let's say we have an input sequence of length $T_x$ and input size $d$.

Then, we define three weight matrices $W_q$, $W_k$, and $W_v$, each of shape $d \times d$.

We use these weight matrices to transform the input sequence into query, key, and value vectors:

- The query ❓ vector $q_i$ for the ith element of the input sequence is obtained by multiplying the input sequence by $W_q$:
    
    $q_i = W_q x_i$
    
- The key 🗝️ vector $k_j$ for the jth element of the input sequence is obtained by multiplying the input sequence by $W_k$:
    
    $k_j = W_k x_j$
    
- The value ⚖️ vector $v_t$ for the th element of the input sequence is obtained by multiplying the input sequence by $W_v$:
    
    $v_t = W_v x_t$
    

where $x_i$, $x_j$, and $x_t$ are the embeddings of the ith, jth, and th elements of the input sequence, respectively.

These vectors are then used to compute the attention scores as the dot product 🔵 of the query and key vectors, which is then scaled by the square root of the dimensionality $d$ to prevent the scores from becoming too large:

$e_{i,j} = \frac{q_i^T k_j}{\sqrt{d}}$

where $q_i$ and $k_j$ are the query and key vectors for the ith and jth elements of the input sequence, respectively.

The attention scores are then normalized using the softmax function 🥎 to get attention weights:

$\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k=1}^{T_x} \exp(e_{i,k})}$

where $T_x$ is the length of the input sequence.

The context vector $c_i$ for the ith element of the input sequence is then computed as the weighted sum ➕ of the value vectors, using the attention weights:

$c_i = \sum_{j=1}^{T_x} \alpha_{i,j} v_j$

where $v_j$ is the value vector for the jth element of the input sequence.

Finally, the context vectors are concatenated 🔗 and passed through a feedforward neural network to obtain the final output.

It represents the model's prediction or the next state of the model. The specific details of the final output would depend on the architecture and task of the deep learning model being used.

However, self-attention is particularly useful in natural language processing tasks, where the input sequence is often a sequence of words or tokens, and the model needs to capture long-range dependencies between them.

Self-attention has been a breakthrough in natural language processing and has been used in many state-of-the-art models such as BERT and GPT-2.

### Differences 🔎

In Generic attention, on the other hand, the input sequence is compared to some additional context to obtain a set of attention weights.

The additional context can be the current state of the model (like decoder outputs), image or another sequence.

For example, 

In image captioning, 📸 the model uses both the image and the previously generated words as additional context to generate a caption. The attention mechanism can then focus on different parts of the image when generating each word of the caption. 

Similarly, in machine translation 🤖, the model can use the previously generated words as additional context when generating the next word. The attention mechanism can then focus on different parts of the source sentence when generating each word of the target sentence.

So, the additional context can be any sequence or vector that is relevant to the task being performed by the model, including the output of the model itself.

The key difference between the two is that in **scaled dot product attention** 🔵, the query, key, and value matrices are all derived from different sources, while in self-attention, they are all derived from the same input sequence.

Scaled dot product attention is commonly used in transformer-based models, while self-attention is commonly used in recurrent neural networks and other models for natural language processing tasks.

This differences will help you better understand the attention mechanism.

## Types of Attention Mechanism

There are several types of attention mechanisms, including:

### Global attention 🌍

In this type of attention, the model considers all the input elements when computing the attention weights. It is also known as hard attention or window-based attention.

For example, in image captioning, global attention can be used to attend to all the regions of the image when generating the corresponding caption.

However, global attention can be computationally expensive when dealing with long input sequences, as it requires computing the attention weights for all the input elements.
    
### Local attention 🏠

In this type of attention, the model only considers a subset of the input elements when computing the attention weights. This subset can be determined based on the current state of the model or other factors. It is also known as soft attention or content-based attention.
    
For example, in machine translation, local attention can be used to attend to a subset of the source sentence when generating the corresponding target sentence.
    
### Multi-head attention 🤹

In this type of attention, the model computes multiple sets of attention weights, each focusing on a different part of the input sequence. 
In multi-head attention, the input is split into multiple heads, each of which is processed using self-attention. The outputs of the multiple heads are then concatenated and passed through a linear layer to produce the final output.
 
This allows the model to capture different aspects of the input sequence simultaneously.
    
One example of a complex task that requires multi-head attention is language modeling 🔮, where the model is trained to predict the next word in a sentence given the previous words. In this task, the model needs to capture both local dependencies between adjacent words and long-range dependencies between distant words. Multi-head attention can be used to capture both types of dependencies by allowing the model to attend to different parts of the input sequence in parallel. This has been demonstrated in models such as GPT-2 and BERT, which have achieved state-of-the-art performance on a wide range of natural language processing tasks.
    
![Fig: As the model generates each word, its attention changes to reflect the relevant parts of the image. “soft” (top row) vs “hard” (bottom row) attention. 
Taken From Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](/assets/2024/September/soft-hard.png)

*Fig: As the model generates each word, its attention changes to reflect the relevant parts of the image. “soft” (top row) vs “hard” (bottom row) attention. 
Taken From Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*

Each type of attention mechanism has its own advantages and disadvantages, and the choice of which one to use depends on the specific task and dataset being used.

## Applications 

Attention has been used in various applications in deep learning, such as machine translation, speech recognition, image captioning, and question answering.

One of the most popular models that use attention is the Transformer, which was introduced in 2017 by Vaswani et al. 

Overall, attention has proven to be a powerful mechanism in deep learning that allows the model to focus on specific parts of the input sequence, improving the performance of various deep learning models.

## References 👏

[Data science](https://towardsdatascience.com/transformers-89034557de14)

[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention → Image captioning using Attention](https://arxiv.org/pdf/1502.03044.pdf)

[Attention is All you Need → Self-Attention, Multi-Head Attention, Transformers](https://arxiv.org/pdf/1706.03762.pdf)



That's it for this blog, hope this was will help you better understand the concepts of deep learning ❤️❤️.