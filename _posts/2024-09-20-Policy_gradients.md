---
layout: post
title:  "Policy Gradients"
date:   2024-10-03 10:00:10 +0530
categories: AI
---

Policy Gradients are a fundamental concept in reinforcement learning, offering a powerful approach to training agents in complex environments. Let's explore this fascinating topic in detail! 🧠

In Reinforcement learning, there are two types of methods value-based methods and policy-based methods.

# Policy 🎮

The algorithm a software agent uses to determine its actions is called its policy. The goal of the agent is to learn a policy that maximizes its reward over time. 🎯

There are several types of policies:

1. **Deterministic policies** 🔒: 
These directly map states to actions
2. **Stochastic policies** 🎲: 
These map states to probability distributions over actions. An action is randomly chosen based on the probability distribution, and it's not necessary for the agent to always choose the action with the highest probability. This type of policy involves some randomness

## Value-Based Methods

Value-based methods focus on learning the value function and *deriving a policy from it*. Some popular algorithms include: Q-Learning, SARSA (State-Action-Reward-State-Action), DQN (Deep Q-Network), Double DQN. 

Check the other post about Q-learning to learn more about value based methods.

[Q-Learning](https://ds-meena.github.io/ai/2024/09/23/Q_learning.html)

## Policy-Based Methods

Policy-based methods refer to any reinforcment learning technicque that *directly learn a policy* without explicitly computing value functions. These can include:

- **Policy Iteration**: A method that alternates between policy evaluation and policy improvement.
- **Evolutionary Algorithms**: Methods that use principles of biological evolution to optimize policies.
- **Policy Gradient Methods**: Techniques that use gradient ascent to optimize the policy directly.

# Policy Gradients 📈

In policy gradients, the agent learns to make decisions based on a policy, which is a mapping from states to actions. The method is particularly useful for problems where the environment is fully observable, and the agent can learn through trial and error. 🧠🔍

The policy gradient algorithm is an iterative process that gradually updates the policy parameters to maximize the expected reward. It improves the policy by playing, using gradient ascent and discounted rewards. The optimization is done using gradient ascent methods, such as stochastic gradient ascent. 🔄💹

Policy gradients have been successfully applied to various domains, including robotics and natural language processing. 🤖💬

### The Fundamental Concept 💡

The core idea of Policy Gradient methods can be summarized in three steps:

1. The agent performs an action based on its current policy 🤖
2. The environment provides feedback in the form of a reward signal 👀
3. The policy is updated to increase the probability of actions that led to high rewards and decrease the probability of actions that led to low rewards 📈📉

## Mathematical Framework 🔢

Let's say we have a simple policy that chooses actions uniformly at random. The expected return for this policy could be calculated as:

$$
E_{τ∼π_θ}[R(τ)] = \frac{1}{N} \sum_{i=1}^N R(τ_i)
$$

Where N is the number of sampled trajectories, and $R(τ_i)$ is the return (sum of rewards) for the $ith$ trajectory.

The Policy Gradient theorem forms the backbone of these methods. The objective is to maximize the expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

Where:

- $J(θ)$ is the expected return
- $θ$ represents the policy parameters
- $τ$ is a trajectory (state-action sequence e.g. `τ = [
    (s0, a0, r0, s1),  # (state, action, reward, next state)
    (s1, a1, r1, s2),
    (s2, a2, r2, s3),
    ...
    (sT, aT, rT, sT+1)
])`
- $π_θ$ is the policy
- $R(τ)$ is the return of a trajectory

The gradient of this objective with respect to the policy parameters is given by:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \lvert s_t) R(\tau) \right]
$$

This equation can be interpreted as follows:

- $∇_θ log π_θ(a_t\lvert s_t)$ is the gradient of the log probability of taking action $a_t$ in state $s_t$
- $R(τ)$ is the return of the trajectory, acting as a weighting factor
- The summation is over all time steps in the trajectory
- The expectation $E$ is taken over all possible trajectories under the current policy

This equation tells us how to adjust our policy parameters θ to increase the expected return. In practice, we often estimate this expectation using a finite number of sampled trajectories.

## Practical Implementation 🖥️

In practice, we often use the following steps to implement Policy Gradient methods:

1. Initialize the policy parameters $θ$ randomly
2. For each episode:
    1. Generate a trajectory $τ$ by following the current policy $π_θ$
    2. Calculate the returns $R(τ)$ for the trajectory
    3. Update the policy parameters using gradient ascent:
        
        $$
        θ ← θ + α ∇_θ J(θ)
        $$
        
        Where α is the learning rate

```python
class PolicyGradient:
    def __init__(self, n_states, n_actions, learning_rate=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        self.optimizer = tf.optimizers.Adam(learning_rate)
    
    # represents our policy π_θ
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='relu', input_shape=(self.n_states,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.n_actions, activation='softmax')
        ])
        return model
    
    # follows stochastic policy π_θ(a_t|s_t)
    def choose_action(self, state):
        state = np.reshape(state, [1, self.n_states])
        probabilities = self.model(state)
        action = np.random.choice(self.n_actions, p=probabilities.numpy()[0])
        return action
    
    # optimize the policy given a sequence of states, actions, rewards (trajectory)
    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        with tf.GradientTape() as tape:
            logits = self.model(states)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)

            # calcualte - log π_θ(a_t|s_t) * R(τ) = -1 * gain
            loss = tf.reduce_mean(neg_log_prob * rewards)
        
        # calculate gradient of -ve gain
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        # apply gradient descent
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
```

This code implements the Policy gradient method:

1. The policy $π_θ$ is represented by the neural network model.
2. The action selection follows the stochastic policy: $π_θ(a_t\lvert s_t$).
3. The train method implements the policy gradient theorem:
    
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t\lvert s_t) R(\tau) \right]$$
    
    Here, tf.nn.sparse_softmax_cross_entropy_with_logits computes $-log π_θ(a_t\lvert s_t)$, and we multiply it by the rewards.

4. The gradient computation and parameter update follow the equation:
    
    $$θ ← θ - α (- ∇_θ J(θ))$$   

    First -ve represents gradient descent and the second -ve represents negative of gradient of gain. That mean it maximizes the rewards.

```python
env = gym.make('Acrobot-v1')
agent = PolicyGradient(n_states=6, n_actions=3)

for episode in range(1000):
    state = env.reset()[0]

    # trajectory of the episode
    states, actions, rewards = [], [], []
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
   
        state = next_state
        
        if done:
            break
    
    # Normalize rewards
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
    
    # Now, optimize the policy using trajectory of the episode
    agent.train(states, actions, rewards)
```

We have implemented the policy gradient of [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) environment. Where it tries to stand up and avoid penalty of -1.

## Advantages and Challenges 🏆🚧

Policy Gradient methods offer several benefits:

- They can handle continuous action spaces effectively
- They can learn stochastic policies, which can be crucial in certain environments
- They can be more stable than value-based methods in some scenarios

However, they also face challenges:

- High variance in gradient estimates, which can lead to unstable learning
- Sample inefficiency, often requiring many interactions with the environment
- Sensitivity to hyperparameter choices

### Advanced Variants 🚀

Several advanced algorithms have been developed to address these challenges:

- **Actor-Critic methods**: Combine policy gradients with value function approximation to reduce variance
- **Trust Region Policy Optimization (TRPO)**: Constrains policy updates to improve stability
- **Proximal Policy Optimization (PPO)**: A simpler and more efficient version of TRPO

Understanding these concepts and their mathematical foundations is crucial for effectively implementing and improving Policy Gradient methods in reinforcement learning applications.

## Conclusion

Policy Gradients offer a powerful and flexible approach to reinforcement learning. By directly optimizing the policy, they can handle complex, high-dimensional action spaces and learn stochastic policies. While they face challenges like sample inefficiency and high variance, ongoing research continues to improve these methods, making them an exciting area of study in the field of AI and machine learning. 🎓🔬

Remember, mastering policy gradients takes practice and experimentation. Happy learning! 🚀🤖