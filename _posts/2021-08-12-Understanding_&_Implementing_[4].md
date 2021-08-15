# Understanding Rainbow: Combining Improvements in Deep Reinforcement Learning

In this we will try to understand the state-of-art Q-learning paper Rainbow DQN. Refer here for [Paper](https://arxiv.org/pdf/1710.02298.pdf).
Refer the implementation [here](https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb).

## Introduction

In recent time their has been many imporvements made in DQN algorithm, in this paper and it has been tried to find the best combination of these improvements to get better resuls.
First we will understand different progressions made in Q-learning. I have stated some of the achievements, that are used in rainbow-Q learning: -

1. Deep Q-Netowrks (DQN; Mnih et al. 2013) -> It uses a combination of Q-learning with CNN and experience replay. It provides ability to learn from images and giving human level performance in many Atari games.

2. Double DQN (2016)-> This algorithm fixes the overestimation bias of Q-learning by decoupling selection and evaluation of bootstrap action.

3. Prioritized Experience replay (2015) -> It replays samples of transitions more oftenly from which there is more to learn. This in turn increases the data efficiency of algorithm.

4. Dueling network architecture (2016) -> This architecture helps to generalize across actions by seperately representing state values and action advantages.

5. Noisy DQN (2017) -> This algorithm uses stochastic network layers for exploration.

6. Distributional Q-learning (2017) -> This algorithm learn a categorical distribution of discounted returns instead of estimating the mean.

7. Multi-Step learning (2016) -> As used in A3C, helps to propagate newly observed rewards faster to earlier visited-states and shifts the bias-variance trade-off.

## Background: -

As being Reinforcement learning algo, this also tries to train an agent to act in partially observable environment and tries to maximize the reward signal.

**Agents and environments:** 

This can be formulated as an Markov decision process (means current state depends on previous states). And this MDP is episodic.

MDP is formulized as tuple = (S, A, T, r, γ)

where,

Set of States, S

Set of actions, A

Stochastic Transition function, ```T(s, a, s') = P[Sₜ₊₁=S'|Sₜ=s, Aₜ=a]```

Reward function, ```r(s, a) = E[Rₜ₊₁|Sₜ = S, Aₜ = a]```

Discount factor, γ ϵ [0, 1]

for at any time step, γₜ = γ except at termination, γₜ = 0.

For Action selection, we use policy 𝛱, it defines probability distribution of given state over actions.

Using discounted return, Gₜ = ⅀ₖ₌0 ∞ k=0 γ(k)t Rt+k+1

Discounted returns represents the dicounted sum of future rewards collected by the agent.

Dicount for a reward k steps in the future, γ⁽ᵏ⁾t = 𝛱ᵢ₌₁ᵏ γₜ₊ᵢ
i.e, Product of discounts before t

Policy, 𝛱
State, v𝛱(s) = E𝛱[Gt | St = s]
State action pair, q𝛱(s, a) = E𝛱[Gt | St = s, At = a]

**Deep RL and DQN:**

In reinforcement learning, we use the following parameters: -

policy, 𝛱(s, a)

q values, q(s, a)

loss = (Rₜ₊₁ + γₜ₊₁ + max a' qθ (St+1, a') - qθ (St, At))²              --------------- (1)

In reinforcement learning, we try to minimize the loss using gradient descent. We backpropagate the loss to parameters θ of online network (that is used to select actions).

## Extensions to DQN

DQN has been an important algorithm in Deep reinforcement learning, but it also has some limitations. I am explaining some of the most useful extensions to it: -

**Double Q-learning:**

It was proposed in 2010 by Van Hasselt, and this extension tries to fix the overestimation problem of DQN. Double Q-learning decouples the selection of the action from its evaluation.

Loss function used : -
loss = (Rₜ₊₁ + γₜ + qθ (St+1, argmax a' qθ (St+1, a')) - qθ (St, At))²

Using this change we can reduce harmful overestimations.

**Prioritized replay:**

Samples transitions with probability Pt

``Pt ∝ |Rt+1 + γt+1 max a' qθ(St+1, a') - qθ(St, At)|ω``

where, ω = shape of distribution (hyperparameter)

This adds new transitions with max priority because their is more to learn from new transitions.

**Dueling network:**

We uses 2 streams of computations in this
a) Value stream
b) advantage stream

action values,
qθ(s, a) = vₙ(fƐ(s)) + aΨ (fƐ(s), a) - ⅀a' aΨ(fƐ(s), a') / Nₐcₜᵢₒₙₛ

where,
fƐ = shared encoder

vₙ = value stream

aΨ = advantage stream

parameters, θ = {Ɛ, n, Ψ}

**Multi-step learning:**

In this we define the truncated n-step return from a given state St

Rt⁽ⁿ⁾ = k=0 ⅀ k=n-1 γt⁽ᵏ⁾ Rₜ₊ₖ₊₁                 --------(2)

loss = (Rₜ⁽ⁿ⁾ + γₜ⁽ⁿ⁾ + max a' qθ (St+n, a') - qθ (St, At))²

**Distributional RL:**

Support,
Zi = vmin  + (i - 1) Vmax - Vmin / Natoms - 1

where, i ϵ {1, ...., Natoms}
z = vector with Natoms ϵ N⁺ atoms

If

time = t

Probability mass on each atom i = Pθⁱ(St, At)

then

distribution, dt = (z, pθ(St, At))

policy, 𝛱*should match target distribution

target distribution ``d't = (Rt+1 + γt+1z, Pθ(St+1, a*t+1)), Dₖₗ(ɸz d't||dt)``       -----------------(3)

Where, ɸz = L2-projection of target distribution onto z,
greedy action w.r.t mean action values, a*t+1 = argmax a qθ(S+1, a)

mean action values, qθ(St, a) = zᵀ pθ (St, a)

**Noisy Nets:**

It proposes a noisy linear layer, which is combination of determinisitc and noisy stream.
It uses below equation in place of y = b + wx

y = (b + Wx) + (bₙₒᵢₛy ⊙ ϵᵇ + (Wₙₒᵢₛy ⊙ ϵw)x),     -------------(4)

where, ϵᵇ and ϵw are random variables
⊙ = element wise product

## The Integrated Agent (or Rainbow)

⊕ Distributional RL ⊕ Multi-step learning

Using (2) and (3) or by combining distribution and multi-step learning we get,

dt⁽ⁿ⁾ = (Rt⁽ⁿ⁾ + γt⁽ⁿ⁾z, pθ(St+n, a*t+n))

and ``loss = Dₖₗ(ɸz dt⁽ⁿ⁾ ||dt)``

⊕ Double Q-learning

We combine above result (of multi-step distribution loss) with double Q-learning.
By using *online network* for selecting bootstrap action a*t+n in state St+n and using*target network* for evaluating the bootstrap action.

⊕ Prioritized Replay

We prioritize the transitions by KL loss, since the algorithm is minimizing this.

``pt ∝ Dₖₗ(ɸz dt⁽ⁿ⁾ ||dt)ʷ``

⊕ Dueling network

fƐ(s) = shared encoder
vₙ [Natoms] = value stream
aΨ [Natoms x Natoms] = advantage stream

Output at atom i and action a = aⁱƐ (fƐ(s), a)

return's ditrubtions:

Pⁱθ(s, a) = exp(vⁱn(ɸ) + aⁱΨ(ɸ, a) - a⁻ⁱΨ(s)) / ⅀j exp(vʲn(ɸ) + aʲΨ(ɸ, a) - a⁻ʲΨ(s))

where, ɸ = fƐ(s)
a⁻ⁱΨ(s) = 1/ Nₐcₜᵢₒₙₛ * ⅀a' aⁱΨ(ɸ, a')

⊕ Noisy Nets

We replace the linear layers with noisy equivalent. In these noisy layers we use factorial Gaussian Noisy.
