---
layout: post
title:  "Q-Learning"
date:   2024-09-23 10:00:10 +0530
categories: AI
---

# Introduction

In this blog, we will learn about the fundamental algorithms used in reinforcement learning. It's not about neural networks but the mathematical algorithms involved in learning.

# Markov Decision Process 🤔

Let's understand the problem, we are trying to solve here. The environment of an agent can be modelled as a Markov decision process, where the agent can choose one of several actions and the transition probabilities depend on the chosen action. 🤖

Our aim is to find an optimal policy for the agent, by following that agent can maximize the rewards earned in the enviornment.

![alt text](/assets/2024/September/markov%20decision%20chain.png)
*Fig: Example of Markov chain  [credit for image](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)*

Let's learn some of the algorithms that are used to find the optimal policy for the agent.

## (State) Value Iteration algorithm 🔄

In this algorithm, we calcualte the state value $V(s)$ for all states.

Optimal state value $V^*(s)$ of any state s, is the sum of all discounted future rewards the agent can expect on average after it reaches a state s, assuming it acts optimally. 🎯

$V\star(s) = max_a \sum_sP(s,a,s\prime)[R(s,a,s\prime)+\gamma.V^*(s\prime)]$  for all s

*Eq: Bellman Optimality Equation*

where, 

- $P(s,a,s’)$ = transition probability from state s to state s’, given that agent chose action a [conditional probability]. 🎲
- $R(s,a,s’)$ = reward the agent gets when it goes from state s to state s’, given that agent chose action a 🏆
- $\gamma$ = discount factor 🈹

If we increase discount factor, we will value the future rewards more.
Bellman optimality equation assumes, that we already have the optimal state value for next state s'. Since, we don't have future value; we update state values iteratively as follows:

1. First initialize all the state value estimates to 0.
2. Iteratively update them using recurrent relation  

	$V_{k+1}(s) \leftarrow  \underset{a}{\max} \underset{s'}{\sum}P(s,a,s') [R(s,a,s') + \gamma.V_k(s')]$ for all s 

	*Eq: Value Iteration algorithm* 🔁

	where

	- $V_k(s)$ = estimated value of state s at the $k^{th}$ iteration

After the Value Iteration algorithm converges, we can derive the optimal policy $π^\star$ for each state s: 🥳

$$
\pi^*(s) = \underset{a}{argmax} \sum_{s'} P(s, a, s')[R(s,a,s') + \gamma V^*(s')]
$$

This means that for each state, the optimal action is the one that maximizes the expected sum of the immediate reward and the discounted optimal value of the next state. 💰

## Q-Value Iteration algorithm 🎲

This algorithm is used to find the optimal state-action values, genreally called Q-values (Quality values). 💡

Optimal Q-value of state-action pair (s, a), $Q^*(s, a)$, is the sum of discounted future rewards the agent can expect on average after it reaches state s and chooses an action a. 💰

It involves following steps:
1. Initialize all Q-values estimates to 0.
2. Then update them using below recurrence relation. 🔄

	$Q_{k+1}(s,a) \leftarrow \underset{s'}{\sum}T(s,a,s')[R(s,a,s')+\gamma.\underset{a'}{max} \space Q_k(s',a')]$

	*Eq: Q-Value Iteration algorithm*

	where:
	- $\underset{a'}{max} \space Q_k(s', a')$ is the maximum Q-value for the next state s' and all possible actions a' at $k_{th}$ iteratin

After the Q-Value Iteration algorithm converges, we can derive the optimal policy $\pi^*(s)$ for each state s.

$$\pi^*(s) = \underset{a}{argmax} \space Q^\star(s,a)$$

That means, when the agent is in state s it should choose the action with the highest Q-Value for that state. 🏆

Let's apply the Q-Value Iteration algorithm to MDP given in above image:

```python
# shape=[s, a, s']  # row - current state, column = action
# s2 to s0 given action a1 transition probability = [2][1][0]
transition_probabilities = [ 
		[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], 
		[[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]], 
		[None, [0.8, 0.1, 0.1], None]
	]

# shape=[s, a, s']
rewards = [  
		[[+10, 0, 0], [0, 0, 0], [0, 0, 0]], 
		[[0, 0, 0], [0, 0, 0], [0, 0, -50]], 
		[[0, 0, 0], [+40, 0, 0], [0, 0, 0]]
	]

# from s0, s1, s2
possible_actions = [[0, 1, 2], [0, 2], [1]]   

# Initialize Q-Values
Q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
	Q_values[state, actions] = 0.0     # 0 for possible actions
	
# Q-Value Iteration algorithm
gamma = 0.90

for iteration in range(50):
	Q_prev = Q_values.copy()
	
	for s in range(3):
		for a in possible_actions[s]:

			Q_values[s, a] = np.sum([transition_probabilities[s][a][sp] * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
				for sp in range(3)])
                        
print(Q_values)
print("Best action for each state: ", np.argmax(Q_values, axis=1))

# [[18.91891892 17.02702702 13.62162162]
#  [ 0.                -inf -4.87971488]
#  [       -inf 50.13365013        -inf]]
# Best action for each state  [0 0 1]
```

Using above algorithm we can find the best policy for the agent.

# Q-Learning 🤖

If you notice in the above MDP diagram, the transition probabilities and rewards are given us in advance. That's not the case in real word 🌍, now comes the role of Q-Learning algorithm. **Q-Learning algorithm** is an adaptation of the Q-Value Iteration algorithm to the situation where the transition probabilities and the rewards are initially unknown.

This algorithm is useful for problems where the environment is fully observable, and the agent can learn by trial and error. Q-learning has been successfully applied to problems such as game playing, robotics, and natural language processing. 🧠🤖

This is an example of **model-free reinforcement learning**, where the transition probabilities and the rewards are initially unknown and agent has to learn these by direct interactions and experiences.

$Q(s,a) \underset {\alpha}{\leftarrow} r + \gamma.\underset{a'}{max} \space Q(s', a')$

*Eq: Q-Learning algorithm*

$ old \underset {\alpha}{\leftarrow} new ⇒ old(1-a) + a*new$ [This is how be interpret the above equation]

## Q-learning algorithm 🧠

1. Initialize the Q-table with arbitrary values for all state-action pairs.
2. Observe the current state.
3. Select an action to take based on the current state and the values in the Q-table. This can be done using an exploration-exploitation strategy such as epsilon-greedy.
4. Take the selected action and observe the reward and the new state. (a, r, s’)
5. Update the Q-value for the state-action pair that was just taken based on the observed reward and the maximum Q-value for the new state.
    
    The Q-learning algorithm uses the following equation to update the Q-value for a state-action pair:
    
    $Q(s,a) {\leftarrow} (1-\alpha)Q(s,a) + \alpha(  r + \gamma.\underset{a'}{max} \space Q(s', a'))$
    
    Where:
    
    - Q(s, a) is the Q-value for state s and action a
    - α is the learning rate, which determines how much the Q-value is updated in each iteration
    - r is the reward received for taking action a in state s
    - γ is the discount factor, which determines the importance of future rewards
    - $\underset{a'}{max} \space Q(s', a')$ is the maximum Q-value for the next state s' and all possible actions a' (maximum future reward estimate)
    - s' is the next state reached after taking action a in state s
    
6. Repeat 🔄 steps 2-5 until the algorithm converges or a maximum number of iterations is reached.

The optimal policy 🏆 can be derived by selecting the action with the highest Q-value for each state as in Q-value Iteration algorithm.

Let’s implement Q-Learning algorithm using open AI gym environment (Taxi-v3). 🚕

```python
env = gym.make('Taxi-v3')

Q_values = np.zeros([env.observation_space.n, env.action_space.n])

# exploration policy
epsilon = 0.1  # Exploration rate

def exploration_policy(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q_values[state])  # Exploit
```

Q-Learning algorithm with learning rate decay: ☢️

```python

# Hyperparameters
alpha0 = 0.1  # Initial learning rate
decay = 0.0001
gamma = 0.99  # Discount factor

num_episodes = 10000
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    
    while not done:
        action = exploration_policy(state)
        next_state, reward, done, _, info = env.step(action)
        
        # Q-learning update
        alpha = alpha0 / (1 + episode * decay)
        
        Q_values[state, action] *= 1 - alpha
        Q_values[state, action] += alpha * (reward + gamma * np.max(Q_values[next_state]))
        
        state = next_state
```

![Fig: The Q-Value Iteration algorithm (left) versus the Q-Learning algorithm (don’t know anything) (right)](/assets/2024/September/Q_learning.png)

*Fig: The Q-Value Iteration algorithm (left) versus the Q-Learning algorithm (don’t know anything) (right) [credit for image](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)*

Obviously, not knowing the transition probabilities or the rewards makes finding the optimal policy significantly harder!

### Advantage

It can learn optimal policies without requiring a model of the environment. (Model-free reinforcement learning algorithm).  Instead, it learns directly from experience by updating the Q-values based on observed rewards and transitions between states.

### Disadvantage

It can be computationally expensive and may require a large amount of data to converge to an optimal solution.

Overall, Q-learning is a powerful technique with many potential applications, but it is important to carefully consider the problem and the available data before choosing a Q-learning approach.

## References

1. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow [Buy here](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)