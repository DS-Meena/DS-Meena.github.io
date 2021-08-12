# Understanding and Implementing Playing Atari using Deep Reinforcement learning

In this we will try to understand and implement the Foundational research paper of deep reinforcement learning [[1]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). Foundational because it was one of the starting papers that revolunize the use of reinformcent learning. This paper gave the most Starting algorithm of deep rl i.e, DQN (using deep networks in the Q-algorithm).
We will try to understand this paper and them implementing together with it.

## Introduction

- This introduces the concept that by using Cnn into Q-learning algorithm ((Model free RL algorithm)) can make huge advancement.
- This uses stochastic gradient descent [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to update weights means that this method uses an iterative method for optimizing an object function.  
- This method uses experience replay and randomly samples previous transitions.  
- This is a **model-free** reinforcement learning algorithm that is, it works without creating an estimate of environment.
- This is an **off-policy** algorithm, because its uses different policies for learning optimal policy (update policy) and taking actions (behavrioral policy) [[off-policy vs on-Policy read here]](https://leimao.github.io/blog/RL-On-Policy-VS-Off-Policy/). It uses greedy startegy as update policy and e-greedy strategy as behavrioral policy.
- To eliminate problems of correlated data and non-stationary, we use an experience replay mechanism which randomly samples previous transitions.

## Background

Considers an agent that interacts with an environment E, and generates a sequence of acitons, observations and rewards. The agent selects action from a set of legal actions, A = {1, ....., K}. The agent only observes the image and additionally get a rewar rt representing change in score.
The reward can be received after many thousand time steps or maybe in next step, this depends on the episode finish (done).

Since the reward depends on the previous images or time steps, hence from the current image, the environment is **partially observable**. Therefore, we used the sequences of actions and observations, st = x1,a1,x2,a2,......,at-1,xt to learn policies and train agent. Each sequence represents a distinc state.

We Can formulate the problem as MDP (Markov decision process).

At time-step t,  
state of environment = St,
action taken by agent = At,
reward function result for (At | St) = Rt,
next state for (At | St) = St+1.

## Experience replay memory

We utilize a technique known as experience replay.

```python
        class ReplayBuffer(object):
            def __init__(self, capacity):
                self.buffer = deque(maxlen = capacity)
```

In this memory we are storing the agent's experience at each time-step.  
We define experience as the tuple of parameters that define transition b/w states of environment i.e, Et = (St, At, Rt, St+1).

```python
            def push(self, state, action, reward, next_state, done):
                self.buffer.append((state, action, reward, next_state, done))
```

To apply Q-learning updates, or minibatch updates, we take a batch of random samples from the experencie replay buffer. We took random so that they are not correlated (not depends on previous transitions) and thereafter reduce variance of the updates, improves performance.

```python
            def sample(self, batch_size):
                state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
                return np.concatenate(state), action, reward, np.concatenate(next_state), done 
```

## Temporal Difference Loss

## Algorithm

Deep Q-learning with Experience Replay

![DQN algo](/images\Deep_RL_R_papers\DQN\DQN_with_experience_replay_algo.png)

1. Initialize replay memory D to capacity N

    ```python
        replay_buffer = ReplayBuffer(size)
    ```

2. Initialzie action-value function Q with random weights

    ```python
        model = DQN(env.observation_space.shape[0], env.action_space.n)

        class DQN(nn.Module):
            def __init__(self, num_inputs, num_actions):
                super(DQN, self).__init__()

                self.layers = nn.Sequential(
                    nn.Linear(num_inputs, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_actions))
    ```

3. For episode = 1, M do:

    Initialize sequence s1 = {x1} and preporcessed sequenced ɸ1 = ɸ(s1)

    ```python
        losses = []
        all_rewards = []
        episode_reward = 0
    ```

    for t = 1, T do
        With probability Ɛ select a random action at
        otherwise slect at = MaxₐQ*(ɸ(st), a; θ)

    ```python
            action = model.act(state, epsilon)

            class DQN (nn.Module):

                def act(self, state, epsilon):
                    if random.random() > epsilon:
                        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
                        q_value = self.layers(state)
                        action = q_value.max(1)[1].action
                    else:
                        action = random.randrange(num_actions)
    ```

    Execute action at in emulator and observe reward rt and image xt+1

    ```python
            next_state, reward, done, _ = env.step(action)
    ```

    Store transition (ɸt, at, rt, ɸₜ₊₁) in D and Set St+1 = st, at, xt+1 and preprocess ɸt+1 = ɸ(st+1)

    ```python
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
    ```

    Sample random minibatch of transitions (ɸj, aj, rj, ɸj+1) from D

    ```python
            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss.item())
            
            def compute_td_loss(batch_size):
                state, action, reward, next_state, done = replay_buffer.sample(batch_size)

                state      = Variable(torch.FloatTensor(np.float32(state)))
                next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
                action     = Variable(torch.LongTensor(action))
                reward     = Variable(torch.FloatTensor(reward))
                done       = Variable(torch.FloatTensor(done))
    ```

    Set yj = {rj                    for terminal ɸj+1

    {rj + γ maxa' (ɸj+1, a'; θ) for non-terminal ɸj+1

    ```python
            next_q_values = model(next_state)
            next_q_value     = next_q_values.max(1)[0]
            expected_q_value = reward + gamma * next_q_value * (1 - done)
            
            q_values = model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    ```

    Perform a gradient descent step on (yi - Q(ɸj, aj;θ))² according to equation 3.

    ![equation 3](/images\Deep_RL_R_papers\DQN\equation_3.png)

    ```python
        optimizer = optim.Adam(model.parameters)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()s
    ```

## Advantages over standard Q-learning: -

1. It provides greater data effiecieny because each stored experience can be used at many places.
2. Due to taking random samples from the experience replay for applying gradient descent, this radnomization breks correlations and redues the variance of updates.
3. Because of off-policy algorithm, this avoids unwanted feedback loops.

I Hope you found this helpful in some way.