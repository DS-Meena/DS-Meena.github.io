---
layout: post
title:  "Transformer of AI World"
date:   2023-06-23 10:00:10 +0530
categories: AI
---

## Introduction 🚀

Are you interested in natural language processing? 🤔 If so, you might have heard of the Transformer, a neural network architecture that has achieved state-of-the-art results in several NLP tasks 🏆, such as machine translation 🌍, text classification 📊, and language modeling 📚. In this blog, we'll explore what the Transformer is, how it works, its advantages and disadvantages, and some of its applications. 🧐

## Definition 📖

The Transformer is a type of neural network architecture that relies heavily on attention mechanisms 👀. It was introduced in 2017 by Vaswani et al. and has since become one of the most popular neural network architectures in the field of natural language processing. 🌟

Its success is largely due to its ability to leverage attention mechanisms, which enable the model to focus on specific parts of the input sequence 🔍. This allows the Transformer to process long sequences of text more efficiently than traditional recurrent neural networks. 🚀

# Working ⚙️

The Transformer architecture is a powerful deep learning model used for natural language processing. It is composed of two main parts: an encoder 🔒 and a decoder 🔓.

The encoder, which is the first part of the model, processes the input sequence by encoding its information into a set of vectors 📊. These vectors capture the meaning of the input sequence and are then passed onto the decoder. 🔄

The decoder is responsible for generating the output sequence, such as a translation or a summary 📝. It does this by decoding the vectors produced by the encoder and generating a new sequence of words based on the encoded information. 🎨

![Fig: Transformer - model architecture [[Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)]](/assets/2024/September/transformers.png)

*Fig: Transformer - model architecture [[Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)]*

## Encoder 🔒

The Encoder is made up of N=6 identical layers, each of which has 2 sub-layers. 🧱🔄

1. **Multi-Head Self-Attention Mechanism 🧠👀**: This mechanism calculates multiple sets of attention weights using self-attention, with each set focusing on a different part of the input sequence.
2. **Feed Forward Network 🔄➡️**: This is a fully connected feed-forward network that is applied to each position separately and identically. For this, two linear transformations are used with ReLU activation between them:
    
    $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$
    
    Here, $x$ is the input vector, $W_1$ and $W_2$ are weight matrices, $b_1$ and $b_2$ are bias vectors, and $\max(0, x)$ is the ReLU activation function. 🧮📊
    
## Decoder 🔓

The decoder is also composed of N=6 identical layers. However, the decoder adds an extra layer: 🔓🔢

1. **Multi-Head Attention 🎭👥**: This layer does not use self-attention, but instead calculates attention weights over the output of the encoder stack. Its keys and values come from the output of the encoder layer, while its queries come from the previous decoder layer.

After each sub-layer, normalization is performed using layer normalization 📏. This process helps maintain stable activations throughout the network. 🧠

#### Normalization

Normalization involves scaling the input features to have zero mean and unit variance, which helps reduce the effect of differences in the scale of the input features. 📊🔍

Layer normalization is defined as: 🧮

$LayerNorm(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

where $x$ is the input vector, $\mu$ and $\sigma$ are the mean and standard deviation of $x$, respectively, $\epsilon$ is a small constant to prevent division by zero, $\odot$ is element-wise multiplication, and $\gamma$ and $\beta$ are learnable scale and shift parameters, respectively. 🔬

In addition, positional encoding is added to the input embeddings of the encoder and decoder stacks. 🎯🔢

#### Positional encoding

The positional encoding is defined as: 🧘🏽

$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$

$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

where $pos$ is the position of the token in the sequence, $i$ is the dimension of the embedding vector, and $d_{model}$ is the total number of dimensions. 📐

These layers allow the model to learn complex patterns and relationships within the input and output sequences. 🕸️ The self-attention layers help the model focus on the most important parts of the input sequence 🔍, while the feed-forward layers help it make predictions based on the encoded information. 🎯

Think in context of training, both input and output are provided. During backpropagation, the weight matrices of the scaled dot-product attention are updated, as well as the parameters of the feedforward neural layers. This process aims to improve the accuracy of the next output and make the attention weights more reasonable. 🔄📈

Overall, the Transformer architecture is a highly effective model for a wide range of natural language processing tasks, including machine translation 🌍 and text summarization 📝. By using multiple layers of self-attention and feed-forward neural networks, it is able to capture complex relationships and patterns within the input and output sequences, making it a powerful tool for language understanding and generation. 🚀🗣️

Let's dive into the attention mechanism used in transformers! 🤿🧠

## Multi-Head Attention 🎭🔍

The multi-head attention operation is like a symphony of smaller attention operations 🎻🎺. These mini-operations work together in parallel, creating a harmonious final output 🎵.

Here's the magical equation for multi-head attention 🧙‍♂️:

$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

Each smaller operation is called an attention "head" 👤. The output of the $i$-th head is calculated like this 🧮:

$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

The weight matrices $W_i^Q$, $W_i^K$, and $W_i^V$ are unique to each $i$-th head, like fingerprints 👆🏼.

![Fig: Multi-Head Attention consists of several attention layers running in parallel. [[Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)]](/assets/2024/September/multi%20head.png)

*Fig: Multi-Head Attention consists of several attention layers running in parallel. [[Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)]*

🔑 Q, 🗝️ K, and 💎 V are the query, key, and value matrices, respectively. The number of attention heads is represented by h 🧠. The final output of the multi-head attention operation is obtained by concatenating the output of each attention head and multiplying it by a weight matrix 🧮 W⁰.

## Scaled Dot-Product Attention 🔍

The scaled dot-product attention operation can be defined as: 🧮

$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

Here, 🔑 Q, 🗝️ K, and 💎 V are the query, key, and value matrices, respectively, and 📏 d_k is the dimension of the key vectors.

Self-attention 🧠👀 is a type of scaled dot-product attention. In self-attention, the query, key, and value vectors are all derived from the same sequence, allowing the model to focus on different parts of the sequence at different times. 🔍🔄

For example, in the case of an encoder, the keys 🗝️, values 💎, and queries 🔑 come from the output of the previous layer of the encoder. 🔁🧱

### Coding View

Here is an example of a Transformer architecture implemented in Python using PyTorch:

```python
import torch
import torch.nn.functional as F

class Transformer(torch.nn.Module):
    def __init__(self, input_size, output_size, num_layers, dropout):
        super(Transformer, self).__init__()

        self.encoder = torch.nn.ModuleList([EncoderLayer(input_size, dropout) for _ in range(num_layers)])
        self.decoder = torch.nn.ModuleList([DecoderLayer(output_size, dropout) for _ in range(num_layers)])
        self.fc = torch.nn.Linear(output_size, output_size)

    def forward(self, input_seq, output_seq):
        enc_output = input_seq

        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output)

        dec_output = output_seq

        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output)

        output = self.fc(dec_output)

        return output

```

The self-attention mechanism 🧠👀 is used in the `EncoderLayer` 🔒 and `DecoderLayer` 🔓 classes, which are used to define the encoder and decoder layers of the Transformer architecture, respectively.

Here are the `EncoderLayer` 🔒 and `DecoderLayer` 🔓 classes used to define the encoder and decoder layers of the Transformer architecture, respectively: 🏗️

```python
class EncoderLayer(torch.nn.Module):
    def __init__(self, input_size, dropout):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(input_size)
        self.feed_forward = FeedForward(input_size)
        self.layer_norm1 = torch.nn.LayerNorm(input_size)
        self.layer_norm2 = torch.nn.LayerNorm(input_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_seq):
        attention_output = self.multi_head_attention(input_seq)
        attention_output = self.dropout(attention_output)
        norm_output = self.layer_norm1(input_seq + attention_output)
        ff_output = self.feed_forward(norm_output)
        ff_output = self.dropout(ff_output)
        output_seq = self.layer_norm2(norm_output + ff_output)

        return output_seq

class DecoderLayer(torch.nn.Module):
    def __init__(self, output_size, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(output_size)
        self.source_attention = MultiHeadAttention(output_size)
        self.feed_forward = FeedForward(output_size)
        self.layer_norm1 = torch.nn.LayerNorm(output_size)
        self.layer_norm2 = torch.nn.LayerNorm(output_size)
        self.layer_norm3 = torch.nn.LayerNorm(output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, output_seq, enc_output):
        self_attention_output = self.self_attention(output_seq)
        self_attention_output = self.dropout(self_attention_output)
        norm_output1 = self.layer_norm1(output_seq + self_attention_output)
        source_attention_output = self.source_attention(norm_output1, enc_output)
        source_attention_output = self.dropout(source_attention_output)
        norm_output2 = self.layer_norm2(norm_output1 + source_attention_output)
        ff_output = self.feed_forward(norm_output2)
        ff_output = self.dropout(ff_output)
        output_seq = self.layer_norm3(norm_output2 + ff_output)

        return output_seq

```

### Advantages and Disadvantages 🌟🚫

One advantage of the Transformer is its ability to process long sequences of text efficiently 🚀. This makes it well-suited for tasks such as machine translation 🌍 and language modeling 📚.

One disadvantage of the Transformer is its high computational cost 💻💰, which can make it difficult to train on large datasets 📊.

### Applications 🛠️

The Transformer has achieved state-of-the-art results in several natural language processing tasks, such as machine translation 🌐, text classification 📋, and language modeling 📝. It has also been used in various applications, such as speech recognition 🎙️, text generation ✍️, and image captioning 🖼️.

### Useful Resources 📚

If you're interested in learning more about the Transformer, here are some useful resources to get you started:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 📄 - The original paper introducing the Transformer architecture.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) 🎨 - A visual guide to the Transformer architecture.
- [Transformers from Scratch](http://peterbloem.nl/blog/transformers) 🔨 - A step-by-step guide to implementing the Transformer from scratch.

I hope you found this blog helpful in understanding the Transformer and its applications in natural language processing 🧠💬. If you have any questions or comments, feel free to leave them below! 💡💬
