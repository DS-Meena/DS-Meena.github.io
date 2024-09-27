---
layout: post
title:  "NLP Techniques to get started with"
date:   2023-04-01 15:08:10 +0530
categories: AI
---

# Introduction

This blog provides an overview of Natural Language Processing (NLP) and describes various NLP techniques such as syntax, n-grams, tokenization, Markov models, text categorization, and semantics. These techniques can be used for information extraction, language identification, machine translation, and sentiment analysis. This blog also includes code snippets and examples to illustrate the implementation of these techniques. In this, we will be using basic python libraries to implement the NLP techniques. 

## NLP

Natural Language Processing (NLP) is a field of AI that focuses on the interaction between human language and computers. It involves developing algorithms and computational models that enable computers to process and analyze human language.

It has many applications like:

- Automatic summarization
- Information extraction
- Language identification
- Machine translation
- Named entity recognition
- Speech recognition
- Text classification
- Word sense disambiguation

## NLP Techniques

NLP Techniques can be broadly classified into 2 categories, syntax based and semantics based. 

## 1. Syntax

Syntax refers to the arrangement of words to make a sentence. It involves using formal grammar rules, such as context-free grammar, to generate sentences in a language. 

### **Formal Grammar**

Formal grammar is a set of rules that define the structure of a language. It is a system of rules for generating sentences in a language. These rules are used to specify how words and phrases can be combined to form sentences in the language. 

Formal grammar can be used to describe both natural and artificial languages. Formal grammar is used to analyze the structure of natural languages and to model the structure of artificial languages like programming languages.

Formal grammar is often used to create programming languages, which are artificial languages used by computers to communicate with humans. These programming languages are designed to be easily understood by both humans and machines.

### **Context-Free Grammar**

A **context-free grammar** (CFG) is a type of formal grammar. 

In a CFG, non-terminal symbols (e.g. N, V, D) represent parts of speech, while terminal symbols (e.g. "she," "saw," "the," and "city") represent actual words. Rules are then applied to generate sentences regardless of context, such as NP → N \| D N, VP → V \| V NP, and S → NP \| VP. We can use the NLTK library to print the parse tree for a sentence.

Example:

Sentence → She saw the city.

Non-terminal symbols → N, V, D

Terminal Symbols → She, saw, the, city.

N (Noun) → She \| City \| car \| Harry \| …… 

V (Verb) → Saw \| ate \| walked \| ….

D (Determiner) → the \| a \| an \| ….. 

P (Preposition) → to \| on \| over \| ….

Adj (Adjective) → blue \| busy \| old \| ….

**Rules**

1. NP → N \| D N
2. VP → V \| V NP
3. S → NP \| VP

Here's an example code snippet using NLTK to generate a parse tree for the sentence "She saw the city.":

```python
import nltk

# Create Grammar
grammar = nltk.CFG.fromstring("""
    S -> NP VP

    AP -> A | A AP
    NP -> N | D NP | AP NP | N PP
    PP -> P NP
    VP -> V | V NP | V NP PP

    A -> "big" | "blue" | "small" | "dry" | "wide"
    D -> "the" | "a" | "an"
    N -> "she" | "city" | "car" | "street" | "dog" | "binoculars"
    P -> "on" | "over" | "before" | "below" | "with"
    V -> "saw" | "walked"
""")

parser = nltk.ChartParser(grammar)

sentence = input("Sentence: ").split()
try:
    for tree in parser.parse(sentence):
        tree.pretty_print()
        tree.draw()
        # break  # print only a single tree
except ValueError:
    print("No parse tree possible.")
```

 

Outputs will we like this

![cfg1-output.png](/assets/2024/September/cfg1-output-768x538.png)

Note that this parse tree shows the syntactic structure of the sentence, with non-terminal symbols representing parts of speech, and terminal symbols representing actual words.

### N-Grams

An N-gram is a contiguous sequence of N items from a sample of text. N-grams can be used to model the probability of the next word in a sentence, given the previous words. 

- Character N-gram: A contiguous sequence of N characters from a sample of text.
- Word N-gram: A contiguous sequence of N words from a sample of text.
- Unigram: A contiguous sequence of 1 item from a sample of text.
- Bigram: A contiguous sequence of 2 items from a sample of text.
- Trigram: A contiguous sequence of 3 items from a sample of text.

Here's an example Python code to generate N-grams:

```python
from collections import Counter

import math
import nltk
import os
import sys

def main():
		"""Calculate the term frequencies of N grams"""

    if len(sys.argv) != 3:
        sys.exit("Usage: python ngrams.py N corpus")
    print("Loading data...")

    n = int(sys.argv[1])
    corpus = load_data(sys.argv[2])

    # Compute n-grams
    ngrams = Counter(nltk.ngrams(corpus, n))

    # Print most common n-grams
    for ngram, freq in ngrams.most_common(10):
        print(f"{freq}: {ngram}")

def load_data(directory):
    contents = []

    # Read all files and extract words
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            contents.extend([
                word.lower() for word in
                nltk.word_tokenize(f.read())
                if any(c.isalpha() for c in word)
            ])
    return contents

if __name__ == "__main__":
    main()
```

The output will show the most common N-grams in the corpus.

![word-n-gram-output.png](/assets/2024/September/word-n-gram-output.png)

### Tokenization

Tokenization is the task of splitting a sequence of characters into pieces (tokens). **Word tokenization** refers to the task of splitting a sequence of characters into words, while **sentence tokenization** refers to the task of splitting a sequence of characters into sentences.

Here's an example Python code using NLTK to perform word and sentence tokenization:

```python
import nltk

sentence = "This is a sentence. This is another sentence."
words = nltk.word_tokenize(sentence)
sentences = nltk.sent_tokenize(sentence)

print(words)
print(sentences)

```

Output:

```
['This', 'is', 'a', 'sentence', '.', 'This', 'is', 'another', 'sentence', '.']
['This is a sentence.', 'This is another sentence.']
```

Tokenizing certain words can be difficult, such as words with contractions like "can't" or hyphenated words like "well-cut". In these cases, special processing may be required to accurately tokenize the words.

Example: "Whatever remains, however improbable, must be the truth."

Tokenized: ["Whatever", "remains,", "however", "improbable", "must", "be", "the", "truth."]

Remove the commas and periods.

Word tokenized vector: ["Whatever", "remains", "however", "improbable", "must", "be", "the", "truth"]

### Markov Models

Markov models are a type of probabilistic model that can be used to generate sequences of data. They are often used in natural language processing to generate text that appears to be similar to human-written text.

A Markov model is a simple model that takes the current state of a system and uses it to predict the next state. In the case of text generation, the system is the sequence of words in a sentence, and the state is the current word. The model uses the current word to predict the next word in the sequence.

Here's a diagram illustrating the basic concept of Markov models:

![Untitled](/assets/2024/September/image-1280x298.png)

Here's an example Python code to generate text using a Markov model:

```python
import markovify
import sys

# Read text from file
if len(sys.argv) != 2:
    sys.exit("Usage: python generator.py sample.txt")
with open(sys.argv[1]) as f:
    text = f.read()

# Train model
text_model = markovify.Text(text)

# Generate sentences
print()
for i in range(5):
    print(text_model.make_sentence())
    print()
```

This code uses the markovify library to train a Markov model on a sample of text from a file. It then generates five sentences using the trained model.

![Untitled](/assets/2024/September/Untitled-14.png)

### Text Categorization

Text categorization refers to the task of assigning predefined categories or labels to a set of documents based on their content.  Text categorization has various applications and can be used in different cases, such as:

- Spam vs. not spam
- Happy vs. sad (sentiment)
- Classify documents into different topics.

Examples:

🙂 My grandson loved it! So much fun.

☹️ The product broke after a few days.

🙂 It's one of the best games I've played in a long time.

☹️ It's kind of cheap and flimsy, not worth it.

### Text Categorization Methods

Text categorization methods include the bag of words model, topic modeling, and term frequency-inverse document frequency (TF-IDF). The **bag of words model** represents text as an unordered collection of words, while **topic modeling** involves discovering the underlying topics in a set of documents. **TF-IDF** ranks the importance of words in a document based on their frequency and rarity across the entire corpus. These methods can be used to classify documents into different categories or labels based on their content, such as sentiment analysis or spam detection.

Here, we will talk about Bag of words model for text categorization.

### 1. Bag of Words Model

The **bag of words model** represents text as an unordered collection of words. In this model, a text is represented as a bag (multiset) of its words, disregarding grammar and word order. 

This model can be used for various tasks such as sentiment analysis. 

In the bag of words model, a text document is represented as a sparse vector (entries represent the frequency of word) of word occurrences, and the **Naive Bayes classifier** can use this representation to classify the document into one of several pre-defined categories.

**Bayes’ Rule (Naive Bayes Classifier)**

Naive Bayes classifier is a machine learning algorithm based on Baye’s rule. It can be used for text classification, and it can be applied to a bag of words model.

 It is called "naive" because it assumes that all the features are independent of each other, which is not always the case. Despite this, the Naive Bayes Classifier has been shown to perform well in various applications.

The rule is based on conditional probability, where the probability of an event happening depends on the occurrence of another event.

The formula for conditional probability is given by

 $P(b\lvert a) = \frac {P(b, a)}{P(a)}$, where $a$ and $b$ are two events. 

On the other hand, the formula for conditional probability with a condition is

 $P(b \lvert a) = \frac {P(b)P(a \lvert b)}{P(a)}$, where $P(a\lvert b)$ is the probability of occurrence of event $a$ given that $b$ has occurred.

In terms of **univariate distribution**, we can use a table to show the probabilities of two events occurring. For example, we can use the following table to show the probabilities of a happy and a sad face:

| 🙂 | ☹️ |
| --- | --- |
| 0.49 | 0.51 |

On the other hand, for **bivariate distribution**, we can use a table to show the probabilities of one event given second event. For example, we can use the following table to show the probabilities of different words given happy or sad face:

| P(word | emotioin) | 🙂 | ☹️ |
| --- | --- | --- |
| my | 0.30 | 0.20 |
| grandson | 0.01 | 0.02 |
| loved | 0.32 | 0.08 |
| it | 0.30 | 0.40 |

It is important to note that these tables are just examples and can be used for various purposes. By using Bayes' Rule and such tables, we can gain valuable insights into various phenomena and make informed decisions.

Here's an example Python code for implementing a Naive Bayes Classifier for sentiment analysis:

```python
import nltk
import os
import sys

def main():

    # Read data from files
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py corpus")
    positives, negatives = load_data(sys.argv[1])
    
    # Create a set of all words
    words = set()
    for document in positives:
        words.update(document)
    for document in negatives:
        words.update(document)

    # Extract features from text
    training = []
    training.extend(generate_features(positives, words, "Positive"))
    training.extend(generate_features(negatives, words, "Negative"))

    # Classify a new sample
    classifier = nltk.NaiveBayesClassifier.train(training)
    s = input("s: ")
    result = (classify(classifier, s, words))
    for key in result.samples():
        print(f"{key}: {result.prob(key):.4f}")

def extract_words(document):
    return set(
        word.lower() for word in nltk.word_tokenize(document)
        if any(c.isalpha() for c in word)
    )

def load_data(directory):
    result = []
    for filename in ["positives.txt", "negatives.txt"]:
        with open(os.path.join(directory, filename)) as f:
            result.append([
                extract_words(line)
                for line in f.read().splitlines()
            ])
    return result

def generate_features(documents, words, label):
    features = []
    for document in documents:
        features.append(({
            word: (word in document)
            for word in words
        }, label))
    return features

def classify(classifier, document, words):
    document_words = extract_words(document)
    features = {
        word: (word in document_words)
        for word in words
    }
    return classifier.prob_classify(features)

if __name__ == "__main__":
    main()
```

This code uses the NLTK library to implement a Naive Bayes Classifier for sentiment analysis. It first loads the data from two text files ("positives.txt" and "negatives.txt"), then extracts the words and generates features for each document. The classifier is then trained on these features, and the user can input a new sentence to be classified as either "positive" or "negative". The output includes the probability of the sentence belonging to each class.

Output:

![Untitled](/assets/2024/September/Untitled-15.png)

### **2. Topic modelling**

**Topic modelling** is a technique used to discover underlying topics in a collection of text documents. It involves using algorithms to identify patterns and group similar words together, resulting in a set of topics that represent the main themes within the text. Topic modelling is often used in text mining and information retrieval applications, such as search engines and recommendation systems. Some popular topic modelling algorithms include Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF).

### 3. TF-IDF

TF-IDF (term frequency-inverse document frequency) is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. It is calculated by multiplying the frequency of a word in a document by the inverse frequency of the word in the corpus. 

This measure is useful for text classification, where it can be used to identify important words or features in a document that can be used to classify it into one or more categories.

Here's an example Python code to calculate TF-IDF scores:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Example documents
documents = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Calculate TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

# Print feature names and scores
feature_names = vectorizer.get_feature_names()
for i in range(len(documents)):
    print(f"Document {i}:")
    for j in np.argsort(tfidf[i].toarray()).flatten()[::-1]:
        print(f"  {feature_names[j]}: {tfidf[i, j]:.4f}")

```

This code uses the TfidfVectorizer class from the scikit-learn library to calculate TF-IDF scores for a set of example documents. The output includes the feature names and scores for each document.

Output:

```
Document 0:
  the: 0.4694
  this: 0.3793
  is: 0.3793
  first: 0.3310
  document: 0.3310

Document 1:
  the: 0.4694
  this: 0.3793
  is: 0.3793
  second: 0.3310
  document: 0.3310

Document 2:
  the: 0.4694
  this: 0.3030
  is: 0.3030
  third: 0.4178
  and: 0.4178
  one: 0.4178

Document 3:
  the: 0.4694
  this: 0.3793
  is: 0.3793
  first: 0.3310
  document: 0.3310

```

In this example, the TF-IDF scores are calculated for the words in each document, and the most important words are identified based on their scores. These scores can be used as features for text classification algorithms, such as Naive Bayes or Support Vector Machines (SVMs).

### **Information retrieval**

Information retrieval is the task of finding relevant information in a collection of documents. This is often done using search engines, which use various techniques to index and search through large volumes of text. Information retrieval techniques include keyword-based search, TF-IDF, and machine learning algorithms.

### TF-IDF

TF-IDF (term frequency-inverse document frequency) is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. It is calculated by multiplying the **term frequency** (number of occurrences of a word in a document) by the **inverse document frequency** (logarithm of the total number of documents divided by the number of documents containing the word). TF-IDF is commonly used for text classification, information retrieval, and other applications where the importance of words needs to be evaluated.

Inverse document frequency is a measure of how common or rare a word is across document.

 $IDF =  log_e(\frac {Total-Documents}{Num-Documents-containing-word })$

$TF-IDF = TF * IDF$

TF-IDF is commonly used in search engines to rank documents based on their relevance to a query. When a user searches for a query, the search engine uses TF-IDF to calculate the relevance of each document in the corpus and returns the most relevant documents to the user. The documents with the highest TF-IDF score for a given query are considered the most relevant to that query. Therefore, TF-IDF is a useful tool for information retrieval, as it allows users to quickly and easily find relevant documents in a large corpus.

Following repo has query search code, this uses the same TF-IDF technique: 
[Query Retrieval using NLP](https://github.com/DS-Meena/Query-Retrieval-using-NLP)

## 2. Semantics

Semantics-based NLP techniques involve understanding the meaning and context of words and phrases in a language. Some techniques include:

- **Semantic Role Labeling**: Identifying the semantic roles (such as agent, patient, and location) of words in a sentence.
- **Word Sense Disambiguation**: Identifying the correct meaning of a word based on the context in which it is used.
- **Named Entity Recognition**: Identifying and classifying named entities (such as people, organizations, and locations) in a text.
- **Sentiment Analysis**: Determining the emotional tone or attitude expressed in a text.
- **Topic Modeling**: Analyzing the words and themes in a corpus of documents to identify topics and patterns.
- **Relation Extraction**: Identifying and extracting relationships between entities in a text.

These techniques are often used in combination with syntax-based techniques to gain a deeper understanding of language.

### Information extraction

Information extraction is a subfield of natural language processing that involves automatically extracting structured information from unstructured or semi-structured documents. This can include identifying and extracting relevant facts, entities, and relationships from text. Some techniques used in information extraction include **named entity recognition**, **relation extraction**, and **text classification**. Information extraction is often used in applications such as search engines, recommender systems, and chatbots to extract relevant information and provide more personalized responses to users.

An example of information extraction would be extracting the names and locations mentioned in a set of news articles. **Named entity recognition** could be used to identify the names of people, organizations, and locations mentioned in the text, while **relation extraction** could be used to identify the relationships between these entities (such as "Barack Obama is the President of the United States"). This information could then be used to create a database of news articles organized by the people, organizations, and locations mentioned in each article, making it easier to search and analyze the content.

Example →  

facebook.txt

..”When Facebook was founded in 2004”.

Amazon.txt

..”When Amazon was founded in 1994”.

When {company} was founded in {year}.

### Word Net

WordNet is a lexical database for the English language that groups words into sets of synonyms called synsets, provides short, general definitions, and records the various semantic relations between these synonym sets. WordNet can be used for various natural language processing tasks, such as word sense disambiguation, information retrieval, and text classification.

WordNet is organized hierarchically, with each synset representing a different concept or sense of a word. Synsets are linked together by various semantic relations, such as **hypernymy** (a word that is a more general or abstract form of another word) and **hyponymy** (a word that is a more specific or concrete form of another word).

![Fig: Hierarchy of [Wordnet]](/assets/2024/September/image-147.png)

*Fig: Hierarchy of [Wordnet](https://analyticsindiamag.com/a-complete-guide-to-using-wordnet-in-nlp-applications/) *

Here is an example Python code to access WordNet using the NLTK library:

```python
import nltk
from nltk.corpus import wordnet as wn

# Find synsets for the word "dog"
synsets = wn.synsets("dog")

# Print the definitions and examples for each synset
for synset in synsets:
    print(f"{synset.name()}: {synset.definition()}")
    for example in synset.examples():
        print(f" - {example}")

# Find hypernyms for the first synset
hypernyms = synsets[0].hypernyms()
print(f"Hypernyms: {[h.name() for h in hypernyms]}")

# Find hyponyms for the first synset
hyponyms = synsets[0].hyponyms()
print(f"Hyponyms: {[h.name() for h in hyponyms]}")

```

This code uses the NLTK library to access WordNet and find the synsets for the word "dog". It then prints the definitions and examples for each synset, as well as the hypernyms and hyponyms for the first synset.

Output:

```markdown
dog.n.01: a member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds
 - the dog barked all night
canis_familiaris.n.01: a domesticated mammal; a familiar spirit kept by a witch or wizard
 - the dog is a familiar of the witch
frump.n.01: a dull unattractive unpleasant girl or woman
 - she got a reputation as a frump
cad.n.01: someone who is morally reprehensible
 - you dirty dog
 - they treated him like a cur
 - the lowest cur listened to his order
 - get out of my sight, you wretch
 - what a dirty dog
Hypernyms: ['canine.n.02', 'domestic_animal.n.01']
Hyponyms: ['Airedale.n.01', 'Bouvier_des_Flandres.n.01', 'Canis_dingo.n.01', 'Canis_familiaris.n.01', 'Canis_minor.n.01', 'Doberman.n.01', 'Great_Pyrenees.n.01', 'Labrador_retriever.n.01', 'Leonberg.n.01', 'Mexican_hairless.n.01', 'Newfoundland.n.01', 'Pekingese.n.01', 'Pomeranian.n.01', 'Poodle.n.01', 'pug.n.01', 'puppy.n.01', 'toy_dog.n.01']

```

In this example, the synsets for the word "dog" are found and printed, along with their definitions and examples. The hypernyms and hyponyms for the first synset are also printed.

### Word Representation

Word representation is the process of representing words in a language as numerical vectors that can be used as input to machine learning models. This is necessary because most machine learning algorithms can only work with numerical data, not text data. There are many techniques for word representation, some of the most popular include:

- **One-hot encoding**: A simple technique where each word is represented as a binary vector, with a 1 in the position corresponding to the word's index in the vocabulary, and 0s elsewhere. This technique is often used as a baseline for comparison with more advanced techniques.
- **Word embeddings**: A family of techniques where each word is represented as a dense vector in a high-dimensional space, with the vector's coordinates encoding the word's semantic and syntactic properties. Word embeddings are often learned using neural network models, such as Word2Vec or GloVe.
- **Distribution encoding**: Distributional representation typically involves representing words as sparse vectors based on their co-occurrence with other words in a corpus. Word embeddings can be generated using distribution representation.
- **Contextualized embeddings**: A recent development in word representation, where each word is represented as a vector that depends on the context in which it appears. Examples of contextualized embeddings include ELMo and BERT.

### **One-hot Representation**

Example: "He wrote a book."

he: [1, 0, 0, 0, ...]

wrote: [0, 1, 0, 0, 0, ...]

a: [0, 0, 1, 0, 0, ...]

book: [0, 0, 0, 1, 0, 0, 0, ...]

Words with similar meanings have similar indices.

For example, "wrote" and "authored" have similar vectors: [0, 1, 0, 0, 0, 0, 0] and [0, 0, 0, 1, 0, 0, 0], respectively.

Similarly, "book" and "novel" have similar vectors: [0, 0, 0, 1, 0, 0, 0, ...] and [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], respectively.

The more similar the meanings of two words, the closer their vectors will be.

### **Distribution Representation**

Here is an example of distribution representation:

“he wrote a book”

he           [-0.34, -0.03, 0.02, -0.18, 0.22, ….]

wrote    [-0.27, 0.40, 0.00, -0.65,  -0.15, ….]

a              [-0.12, -0.25, 0.29, -0.09, 0.40, ….]

book      [-0.23, -0.16, -0.05, -0.57, …….]

Each value represents a specific meaning, with similar meanings having similar values.

### **Word Embeddings**

Word embeddings are dense vector representation of words in a high-dimensional space, where words with similar meanings are located closer together. [means less dimensions are required compared to one-hot representation]  

Word2Vec is a neural network architecture that learns these word vectors, known as word embeddings. 

**Train model to capture Relationship between words**

There are many existing mathematical techniques for capturing the important structure of a high-dimensional space in a low dimensional space.

For example, [principal component analysis](https://wikipedia.org/wiki/Principal_component_analysis) (PCA) has been used to create word embeddings. Given a set of instances like bag of words vectors, PCA tries to find highly correlated dimensions that can be collapsed into a single dimension.

"Word2vec is used to learn word embeddings" is a more commonly used phrasing in the NLP community, as it emphasizes the fact that word embeddings are learned from data rather than generated from scratch.

Here's an example Python code to generate word vectors using Word2Vec:

```python
from gensim.models import Word2Vec

# Example sentences
sentences = [
    "I love to play football",
    "She enjoys playing chess",
    "He hates swimming",
    "They like to dance",
    "We prefer reading books",
    "You dislike running",
]

# Preprocess sentences (remove punctuation and lowercase)
sentences = [s.lower().replace(".", "").split() for s in sentences]

# Word2Vec model learn the word embeddings
model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

# Print word vectors for some example words
print("Word vector for 'football':", model.wv["football"])
print("Word vector for 'reading':", model.wv["reading"])
print("Word vector for 'swimming':", model.wv["swimming"])

```

In the above below, the Word2Vec model is learning word embeddings for the specific sentences provided. The Word2Vec model learns these embeddings by predicting the context of each word in the sentence, based on the words that surround it. By doing so, the model learns to represent words that have similar contexts or meanings, with similar vector representations in the embedding space. 

Output:

```
Word vector for 'football': [ 0.00961521 -0.01828998 -0.00957083  0.00582886 -0.01004235  0.00128278
  0.00485666 -0.00454647 -0.00431244 -0.00078377 -0.00065795 -0.01995155
  ...
Word vector for 'reading': [ 0.00572616 -0.00882261 -0.00071661  0.01290313 -0.01122897 -0.00116539
 -0.00823871 -0.00643388 -0.0110871  -0.00170786 -0.0056673  -0.01761658
  ...
Word vector for 'swimming': [ 0.01692077  0.00346931  0.00369379 -0.00355725  0.00599522 -0.01022142
 -0.00196455 -0.0164674   0.00604031 -0.00220976 -0.01128828  0.00769941
  ...

```

In this example, word vectors are generated for the words in the example sentences using the Word2Vec model. The word vectors are 100-dimensional, and some example vectors are printed for the words "football", "reading", and "swimming".

### Skip-gram architecture

The **skip-gram architecture** is a neural network model used for generating word embeddings. It is similar to the Word2Vec model, but instead of predicting a target word given a context window, it predicts the context window given a target word. This is useful for generating high-quality embeddings for rare or infrequent words, which may not appear frequently enough in a corpus to be accurately predicted using a context window.

![Figure: Skip-gram architecture](/assets/2024/September/1_SR6l59udY05_bUICAjb6-w.png)

*Figure: Skip-gram architecture*

Ex → Given dinner might generate breakfast, lunch, etc. Given book might generate memoir, novel, etc.

I hope you learned something useful in this blog ❤️❤️.
