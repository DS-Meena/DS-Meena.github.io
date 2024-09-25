---
layout: post
title:  "Policy Gradients"
date:   3024-09-20 10:00:10 +0530
categories: AI
---

In this blog, we will discuss about the policy and policy gradients in reinforcement learning.

# Policy 🎮

The algorithm a software agent uses to determine it’s actions is called it’s policy. The goal of the agent is to learn a policy that maximizes its reward over time. 🎯

There are several types of policies:

1. Deterministic policies 🔒: 
These directly map states to actions
2. Stochastic policies 🎲: 
These map states to probability distributions over actions. An action is randomly chosen based on the probability distribution, and it's not necessary for the agent to always choose the action with the highest probability. This type of policy involves some randomness.

This distinction between deterministic and stochastic policies is an important concept in reinforcement learning and policy-based methods. 🧠

# Policy-based Methods 🎮

In value-based methods, we learn the value function and derive a policy from it (e.g., in Q-learning we choose the action with the highest Q-value). In policy-based methods like policy gradients, we directly optimize the policy without explicitly computing a value function. 🎯

However, some advanced methods, such as actor-critic algorithms, combine both approaches by learning both a value function and a policy simultaneously. 🤖

# Policy Gradients 📈

In policy gradients, the agent learns to make decisions based on a policy, which is a mapping from states to actions. The method is particularly useful for problems where the environment is fully observable, and the agent can learn through trial and error. 🧠🔍

The policy gradient algorithm is an iterative process that gradually updates the policy parameters to maximize the expected reward. It improves the policy by playing, using gradient ascent and discounted rewards. The optimization is done using gradient ascent methods, such as stochastic gradient ascent. 💹

Policy gradients have been successfully applied to various domains, including robotics and natural language processing. 🤖💬

## Algorithm

One popular class of PG algorithms called REINFORCE algorithms (1992). Common variant:

1. Let the neural network play the game several times and keep calculating the gradients that would make the chosen action more likely - but don’t apply yet.
2. Once you have run several episodes, compute each action’s advantage (+ve for good, -ve for bad).
3. Multiply each gradient vector by the corresponding action’s advantage.
4. Finally compute the mean of all resulting gradient vectors, and use it to perform a Gradient Descent step.