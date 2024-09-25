---
layout: post
title:  "Gradient Descent"
date:   2024-08-03 15:08:10 +0530
categories: AI
---

In this blog post, we will understand the underlying algorithm for training neural networks. We will discuss gradient descent, the most basic step of training any model, and backpropagation.

# Gradient Descent

The gradient is, simply a row vector of a function’s partial derivatives. It represents the direction and rate of the steepest increase of a function. 

Example: gradient $\nabla f$ of the function $f(a, b, c)  = ab^2 + 2c^3$, where the variables in order, are a, b, and c:

$\nabla f = (\frac{\partial f}{\partial a} \frac{\partial f}{\partial b} \frac{\partial f}{\partial c}) = (b^2 \space\space\space\space 2ab \space\space\space\space 6c^2)$

In basic calculus, with a simple algebraic function such as a polynomial, a standard optimization process is to:

1. take the derivative of the function
2. set the derivative equal to 0, and then
3. solve for the parameters (inputs) that satisfy this equation.

Since, ANNs functions are very complicated, solving for 0 is not possible. Thus, **heuristic methods are often used** (trail & error).

Gradient descent is a heuristic method that starts at a random point and iteratively moves in the direction (hence “gradient”) that decreases (hence “descent”) the function that we want to minimize, which is usually a cost function. With enough of these steps in the decreasing direction, a local of these steps in the decreasing direction, a local minimum can theoretically be reached. 

Colloquially, think of it as playing a game of “hot” and “cold” until the improvement becomes negligible. 

![nse-6188589518431236078-512387566.png](/assets/2024/September/gradient%20descent.png)

The gradient indicates the direction of steepest ascent of the function, and its negative is the direction of steepest descent.

### Gradient Descent Algorithm

1. Initialize the weights and biases of the neural network with random values.
2. Do the following until the cost stops improving or a fixed number of iterations:
    1. Calculate cost function value with current parameters.
    2. Calculate the gradient of the cost function w.r.t to its parameters.
    3. Update the parameters by taking a small step in the opposite direction of the gradient.

```python
def gradient_descent(point, step_size, threshold):
	cost = f(point)  
	new_point = point - step_size * gradient(point)  # -ve for descent
	new_cost = f(new_point)
	
	# if doesn't improve cost
	if abs(new_cost - cost) < threshold:
		return value
	
	# go to new point
	return gradient_descent(new_point, step_size, threshold)
```

Let’s look at an example of how we update the parameters:

Let’s say we function $f(x, y) = 2 + 3x^2 + 4y^2$, and we try to do a single step of gradient descent. Find the next set of parameters if we start at (10, 10) with step size = 0.5?

Solution: We calculate gradient of this function, gradient $\nabla f = (\frac{\partial f}{\partial x} \frac{\partial f}{\partial y}) = (6x, 8y)$

We know that,

$\text{new point} = point - \text{step size} * gradient(point)$

$\text{new point} = (10, 10) - 0.5 * (12, 16) ⇒ (10, 10) - (6, 8) ⇒ (4, 2)$

Here, we moved in the direction opposite to the vector that is one-half the gradient at (10, 10).

A more scary way to write the above function and equation of gradient descent:
$w_{j+1} = w_j - \eta \nabla Q(w_j)$

where,

$w$ is a vector of parameters we are optimizing over

$Q$ is our cost function

$\eta$ learning parameter

$\nabla Q$ gives direction of steepest ascent, and $- \nabla Q$ points to direction of steepest descent.

Once you get to the bottom of the bowl, $\nabla Q = 0$, the algorithm terminates.

![Fig: Gradient Descent pitfalls](/assets/2024/September/gradient%20descent%20pitfalls.png)

*Fig: Gradient Descent pitfalls*

Gradient descent has some pitfalls, like it might get stuck at a local minimum, or the gradient might be much lower, e.g., on a plateau. Because of this, it might take more time to train.

Gradient descent is a key component of training neural networks, and understanding its properties and variants is important for building effective models.

## Types of Gradient Descent

There are several variants of gradient descent, including batch gradient descent, stochastic gradient descent, and mini-batch gradient descent. These variants differ in the number of examples used to compute the gradient at each iteration and can have different convergence rates and computational requirements.

Some of the Gradient Descent are explained below:

### 1. Batch Gradient Descent
    
Batch gradient descent (BGD) computes the gradient of the cost function with respect to the parameters for the entire training set at each iteration. The weights and biases are then updated based on this gradient. 

gradient is calculated using all training examples.

Example → 

$Q(w) = \frac{1}{N}   \sum_{i=1}^N Q_i(w)$    = cost is calculated using all training examples

$\nabla Q = \frac{1}{N} \sum_{i=1}^N \nabla Q_i$           = gradient is calculated using all training examples.

Gradient itself contains all parameters, here we are talking about examples.

how much the cost function will change if you change $\theta_j$ just a little bit. This is called *partial derivative.*

$\frac{\partial}{\partial\theta_j}  MSE(\theta) = \frac{2}{m} \sum_{i=1}^{m} (\theta^Tx^{(i)} - y^{(i)})x_j^{(i)}$

![Fig: Gradient vector of the cost function](/assets/2024/September/gradient%20vector%20of%20cost.png)

*Fig: Gradient vector of the cost function*

Once you have the gradient vector, which points uphill, just go in the opposite direction to go downhill. This means subtracting $\Delta_{\theta}MSE(\theta)$ from $\theta$ (it’s like going one unit back).

$\theta^{\text{next step}} = \theta - \eta\Delta_{\theta}MSE(\theta)$

Eq: Gradient Descent step

BGD can be computationally expensive, especially for large datasets, but it can converge quickly and reach a global minimum of the cost function.
    
### 2. Stochastic Gradient Descent
    
Stochastic gradient descent (SGD) updates the weights and biases based on the gradient of the cost function with respect to the parameters for a single training example at each iteration. The gradient is therefore noisy and may not be representative of the overall gradient. 

gradient is calculated using one example (as we have been doing in perceptron)

If N is extremely large, computing

$\nabla Q = \frac{1}{N} \sum_{i=1}^N \nabla Q_i$ 

and evaluating all N functions at w may be very time-consuming.

Stochastic Gradient Descent (SGD) is called "stochastic" because it uses a random sample. This randomness helps SGD to avoid local minima and converge faster.
    
### 3. Mini-Batch Gradient Descent
    
Mini-batch gradient descent (MBGD) is a compromise between BGD and SGD. It computes the gradient of the cost function with respect to the parameters for a small batch of training examples at each iteration. 

MBGD can be more computationally efficient than BGD and less noisy than SGD, and it can converge quickly with appropriate batch sizes.
    

The choice of gradient descent algorithm depends on the specific problem and the available computational resources.

# Backpropagation

If gradient descent was a single step to optimize the weights, then backpropagation is the complete algorithm to train neural networks.
Let’s understand the intuition behind backpropagation with an example. Suppose a we are predicting the affinity of a person being male athlete on basis of purchases. We have one hidden layer with two neurons which correspond to the likelihood of being male and related to sports. For input X, we predict some output and later get to know that the input was male sports shoes. Now, how should our neural network update the weight of item X in predicting a male if the weight is currently 0? It should increase the weight.

**Backpropagation** (backward propagation of errors) is an algorithm for supervised learning of ANN using gradient descent. It works by computing the gradient of the cost function with respect to the network's parameters, and then using this gradient to update the parameters using gradient descent. It is a generalization of the delta rule for perceptron's to multilayer feedforward neural networks.

The backpropagation algorithm consists of two main steps:

### 1. Forward pass

The input is fed through the network, and the output is computed for each layer using the current values of the weights and biases.

It also preserves the intermediate results since they are needed for the backward pass.

### 2. Backward pass

The gradient of the cost function with respect to the weights and biases of each layer is computed using the previously computed gradients and the output of the layer. Finally, the weights and biases are updated using the computed gradients and gradient descent.

Let’s get to the proof:

Define for each neuron j in layer l the output $o_j^{(l)}$ such that 

$o_j^{(l)} = \phi(a_j^{(l)}) = \phi \bigg ( \sum_{k=1}^n w_{kj}^{(l)} o_k^{l-1} \bigg),$

where, 

$o_k^{(l-1)}$ neuron output from previous layer

$w_{kj}^{(l)}$ is weight on synapse from k to j (previous layer k neuron to current layer j neuron)

$a_j^{(l)}$ the “activation” of the neuron

bias is omitted

To find how any error function $E$ (usually mean squared error) changes with respect to a weight $w_{ij}^{(l)}$, we apply the chain rule 

$\frac {\partial E}{\partial w_{ij}^{(l)}} = \frac {\partial E}{\partial o_j^{(l)}}\frac {\partial o_j^{(l)}}{\partial a_j^{(l)}} \frac {\partial a_j^{(l)}}{\partial w_{ij}^{(l)}}$

Derivation for Gradient of error function w.r.t to parameters:

$\frac {\partial a_j^{(l)}}{\partial w_{ij}^{(l)}} = o_i^{(l-1)}$

For other terms, we can derive the identify by using the chain rule to write 

$\delta_j^{(l)} = \frac {\partial E}{\partial o_j^{(l)}}\frac {\partial o_j^{(l)}}{\partial a_j^{(l)}}$  

 $\qquad = \frac {\partial o_j^{(l)}}{\partial a_j^{(l)}} \bigg ( \frac {\partial E}{\partial o_j^{(l)}} \bigg)$

$\qquad = \phi' \bigg(a_j^{(l)} \bigg) \sum_m \frac{\partial E}{\partial o_m^{l+1}} \frac{\partial o_m^{l+1}}{\partial a_m^{l+1}} \frac{\partial a_m^{l+1}}{\partial o_j^{l}}$   divide and multiply by $\partial o_m^{l+1}$,  $\partial a_m^{l+1}$

$\qquad = \phi' \bigg(a_j^{(l)} \bigg) \sum_m \delta_m^{l+1} w_{jm}^l$              neuron j from current layer sends signals to next layer neuron m.

Note that this is our backpropagation formula. This allows us to compute previous layers of $\delta_j$ by later layers recursively — this is where backpropagation comes from. We can compute $\delta_j$ directly if $j$ is an output layer, so this process eventually terminates.

If we combine both of our results we can calculate gradient of Error w.r.t parameters:

$\frac {\partial E}{\partial w_{ij}^{(l)}}  = \phi' \big(a_j^{(l)} \big) \sum_m w_{jm}^{(l)} \delta_m^{(l+1)} \qquad o_i^{(l-1)},$

time complexity: $O(mn) = O(W)$

where m is the number of neurons in this layer and n is the number of neurons in next layer.

w = number of synapses in the network

The algorithm is called backpropagation because the gradient is computed backwards through the network, starting from the output layer and working backwards towards the input layer. This allows the algorithm to efficiently compute the gradient for each parameter in the network versus the naive approach of calculating the gradient of each layer separately, which can be used to update the weights and biases.

## Vanishing Gradient Problem

Backpropagation is a powerful algorithm that has enabled the training of deep neural networks with many layers. However, it can suffer from the vanishing gradient problem, where the gradient becomes very small as it is propagated backwards through the network, making it difficult to update the weights and biases of the early layers in the network. 

### Why Gradient keep shrinking

The gradient at each layer is the product of the gradients of the subsequent layers multiplied by the gradient of the current layer. If the gradients of the subsequent layers are small, then the gradient of the current layer will also be small, which can make it difficult to update the weights and biases of the early layers in the network. This is known as the vanishing gradient problem.

This happens because of the activation function, because the activation function compresses the entire real numbers into a small range. You multiply few activation results and gradient becomes too small.

Example → Sigmoid always gives output between [0, 1], If we use sigmoid while calculating gradient’s for previous layers. It keep getting smaller.

This problem has been addressed through the use of activation functions such as ReLU and the development of more advanced optimization algorithms such as Adam and RMSProp.

Sometimes, a smooth approximation to this function is used: $f(x)=ln⁡(1+e^x),$ which is called the **softplus function**.

That's if for this blog, hope this was worth your time. 🥰

## References

1. [Brilliant.org](www.brilliant.org)
2. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow [Buy here](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)