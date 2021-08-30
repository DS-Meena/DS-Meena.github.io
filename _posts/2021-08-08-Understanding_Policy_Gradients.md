# Understanding Policy Gradients

In this, we will discuss the policy gradients methods. Policy Gradients methods comes under  model-Free reinforcement learning and are based on policy based techniques.

## Policy Gradients: -

Using policy gradients, we try to improe our policy. In this we use take gradients of the policy and steepest ascent method, and try to imporve our policy's score.

## Reward and return: -

Reward given as:

    rt = R(st, at, st+1)

Returns are of 2 types:

1. **Finite-horizon undiscounted return** 
It is the sum of rewards obtained in a fixed window of steps.

        Gt = ⅀ᵀₜ₌0 rt

2. **Infinite-horizon discounted return** -> It is the sum of all rewards ever obtained by the agent and discounted by how far in the future they are obtained (adding uncertainity of rewards in future, γ).

        Gt = ₜ₌0⅀∞ γᵗ rt

## Value functions: -

**1. State-Value function:**

The state value function provides us the expected return when starting from a given state s and then following policy 𝛱.

    V𝛱(s) = 𝔼𝛱[Gₜ | Sₜ = s]

Optimal State-value function: -

    V*(s) = max𝛱 𝔼[Gₜ | Sₜ = s]

Value function or state Value function represents how good a state is for the agent. and it depends on what policy the agent follows.

It's value depends on the Policy(𝛱) which the agent follows.

**2. Action-Value function:**

The Action-Value function provies us the expected return starting from state s, taking action a, and then following policy 𝛱.

    q𝛱(s, a) = 𝔼𝛱[Gₜ | Sₜ = s, Aₜ = a]

Optimal Action-value function: -

    Q*(s, a) = max𝛱 𝔼[Gₜ | Sₜ = s, Aₜ = a]

## Policy Based Methods: -

In this methods we use policy function that maps state to action without using a value function. These methods tries to optimize the policy functions for the optimization persepective.

Advantages of Using Policy based techniques: -

1. Convergence

    Value based methods have big oscillations while training because chocie of actions changes dramatically with slight change in action values.
 
    While, Policy based methods have better convergence smoothly than value-based methods because they follow the gradient of policy to find best parameters. It will converge either at local maximum (worst case) or at global maximum (best case).

2. High dimensional action spaces.

    What Q-learning does is that it assigns a future reward for each possible action, at each time step, given current state. But if there are too many chocies, like in driving car, it will be inefficient to calculate reward for each possible action.

    But it does not happend with Policy-based methods. Policy-based methods ajust the parameters directly and estimate the action with maximum reward at each time-step.

3. Stochastic policies

    Policy gradient can learn a stochastic policy while value functions can't. This gives two more advantages : -

    a) No need to implement exploration/exploitation trade off. A stochastic policy allows our states to explore because it outputs a probablity distribution of actions for given state.

    b) Removes problem of perceptual aliasing.

Disadvantage of using policy based techniques: -

Offently, policy gradients converges on local maximum instead of global maximum.

## Solution to get Global optimum policy: -

As we know, In policy based methods, we have policy 𝛱 with parameters θ.

    𝛱θ(a|s) = P[a|s]

To optimize our policy we use score function J(θ) by finding best parameters θ.

    J(θ) = E𝛱θ[⅀ γr]

Score function tells us how good is our policy and Using Policy gradient ascent we try to find best policy parameters.

Step 1: Get Policy Score Function J(θ)

This function calculates the expected reward of policy.
We have 3 ways to calculate policy score and choice depends on environment and objective.

    a) In episodic environment,

    J₁(θ) = E𝛱[G1 = R1 + γR2 + γ²R3 + ....]        -- Cumulative discounted rewards starting at start state
          = E𝛱(V(s1))      ---- Value of state 1
    We only try to maximize the return of episode, because it will be optimal policy.

    b) In continuous environment, we use average value instead (because we can't rely on start state).

    Average state score
    Jₐᵥgᵥ(θ) = E𝛱(V(s)) = ⅀ d(s)V(s)

    where d(s) = N(s)/ ⅀ₛ'N(s') 
                = No of occurances of the state/ Total number of occurances of all states

    Average reward per time step
    JₐᵥR(θ) = E𝛱(r) = ⅀s d(s) ⅀a 𝛱θ(s, a) Rₛᵃ

    where,
    ⅀s d(s) = probability of being in state s
    ⅀a 𝛱θ(s, a) = probability of taking action a from state s under policy 𝛱
    Rₛᵃ = reward for doing action a in state s. 

Step 2: Find optimal parameters (θ) : Policy Gradient ascent

Gradient descent - helps in finding the minima of a function (where the function will give its minimum value)

gradient - derivative
descent - minima

By not setting df/dx = 0, but
finding minima by manipulating symbols, gradient descent approximates the solution with numbers.

In gradient descent, we move towards the steepest decrease in function
and tries to decrease the value of a function (like loss function).

In gradient ascent, we move towards the steepest increase in function and tries to increase the value of a function (like score function).

--------------------

Idea - find gradient of policy 𝛱 that updates the parameters in direction of greatest increase and iterate.

``Policy: 𝛱θ``

``Objective function : J(θ)``

``Gradient: ∇θ J(θ)``

``Update: θ ← θ + ɑ∇θ J(θ)``

We will maximize the score function (means optimize the policy) by finding best parameters (θ*).

``θ* = argmax θ 𝔼𝛱θ [t⅀ R(st, at)]``

``Policy gradient: E𝛱 [∇θ (log𝛱(s, a, θ)) R(τ)]``

``Update rule : Δθ = ɑ * ∇θ (log𝛱 (s, a, θ)) R(τ)``

where,
𝛱(s, a, θ) = policy function
R(τ) = score function
Δθ = change in parameters
ɑ = Learning rate.

## Reference:-

Learn about Value functions [here](https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa)

Learn about Policy based methods [here](https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/)

Gradient descent [here](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/what-is-gradient-descent)