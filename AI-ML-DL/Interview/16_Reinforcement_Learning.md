# Reinforcement Learning

## Q1: What is the reinforcement learning problem formulation?

**A1:** Reinforcement learning formulates learning as an agent interacting with an environment to maximize cumulative reward. The agent observes states, takes actions, receives rewards, and transitions to new states. The goal is to learn an optimal policy that maps states to actions to maximize expected long-term reward. This differs from supervised learning as there are no labeled examples, only reward signals. The agent must balance exploration of unknown actions with exploitation of known good actions.

## Q2: Explain the components of a Markov Decision Process (MDP).

**A2:** An MDP consists of five components: states (S), actions (A), transition probabilities (P), reward function (R), and discount factor (γ). States represent the current situation of the environment. Actions are choices available to the agent. Transition probabilities define the likelihood of moving from one state to another given an action. The reward function provides immediate feedback for state-action pairs. The discount factor balances immediate versus future rewards, with values between 0 and 1.

## Q3: What is the value function in reinforcement learning?

**A3:** The value function V(s) estimates the expected cumulative reward starting from state s and following a policy π. It represents the long-term value of being in a state. The value function satisfies the Bellman equation, which expresses the value recursively as immediate reward plus discounted future value. For optimal policy π*, V*(s) gives the maximum possible expected return. Value functions help evaluate policies and guide learning by indicating which states are more valuable.

## Q4: What is the Q-function and how does it differ from the value function?

**A4:** The Q-function Q(s,a) estimates the expected cumulative reward of taking action a in state s and then following policy π. Unlike the value function V(s) which only depends on states, Q(s,a) evaluates state-action pairs. Q*(s,a) represents the optimal Q-function, giving the maximum expected return for each state-action pair. The Q-function is fundamental to value-based RL methods like Q-learning, as it directly guides action selection by choosing actions with highest Q-values.

## Q5: Explain the Bellman equation and its significance.

**A5:** The Bellman equation expresses the value function recursively, breaking down the value of a state into immediate reward plus discounted future value. For value function: V(s) = R(s) + γΣ P(s'|s,π(s))V(s'). For Q-function: Q(s,a) = R(s,a) + γΣ P(s'|s,a)max Q(s',a'). This recursive structure enables dynamic programming solutions and iterative value estimation. The Bellman equation is fundamental to temporal difference learning, where values are updated based on bootstrapped estimates rather than complete episodes.

## Q6: What is the exploration vs exploitation dilemma?

**A6:** Exploration involves trying new actions to discover potentially better strategies, while exploitation uses known good actions to maximize immediate reward. Pure exploitation may lead to suboptimal policies if better actions remain unexplored. Pure exploration wastes time on poor actions. Effective RL algorithms balance this trade-off, exploring sufficiently to find optimal policies while exploiting knowledge to achieve good performance. Common strategies include epsilon-greedy, UCB, and Thompson sampling.

## Q7: Explain the epsilon-greedy strategy.

**A7:** Epsilon-greedy balances exploration and exploitation by selecting the best-known action with probability (1-ε) and a random action with probability ε. The parameter ε typically starts high for exploration and decays over time. This ensures the agent explores all actions while gradually shifting to exploitation. A common decay schedule is exponential or linear reduction of ε. While simple and effective, epsilon-greedy treats all non-optimal actions equally, ignoring uncertainty estimates.

## Q8: How does the Q-learning algorithm work?

**A8:** Q-learning is an off-policy temporal difference algorithm that learns optimal Q-values directly. It updates Q(s,a) using: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)], where α is the learning rate. The algorithm uses the maximum Q-value of the next state regardless of the policy being followed, making it off-policy. Q-learning converges to optimal Q* under conditions of sufficient exploration and appropriate learning rate decay. It's model-free, requiring only samples of experience.

## Q9: What is SARSA and how does it differ from Q-learning?

**A9:** SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference algorithm. It updates Q-values using: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)], where a' is the action actually taken in state s' under the current policy. Unlike Q-learning which uses max Q(s',a'), SARSA uses the Q-value of the action actually taken. This makes SARSA on-policy, learning the value of the policy being followed. SARSA is more conservative and safer for online learning, while Q-learning learns optimal policy directly.

## Q10: Explain Deep Q-Network (DQN) and its key innovations.

**A10:** DQN combines Q-learning with deep neural networks to handle high-dimensional state spaces. Key innovations include experience replay, storing transitions in a buffer and sampling random batches to break correlation. The target network provides stable Q-targets by using a separate network updated periodically. DQN uses convolutional networks for image inputs and fully connected layers for Q-value estimation. These techniques stabilize training and enable RL in complex environments like Atari games.

## Q11: What is experience replay and why is it important?

**A11:** Experience replay stores agent experiences (state, action, reward, next state) in a replay buffer and samples random batches for training. This breaks correlation between consecutive samples that would cause instability in neural network training. By mixing experiences from different time steps, the network learns from diverse situations. Experience replay improves sample efficiency by reusing experiences multiple times. The buffer size and sampling strategy significantly impact learning performance.

## Q12: Explain the target network in DQN.

**A12:** The target network is a copy of the main Q-network updated less frequently (every C steps). It provides stable Q-targets for the Bellman update: y = r + γ max Q_target(s',a'). Without a target network, Q-targets change constantly as the network updates, causing instability. The target network freezes Q-values temporarily, creating stationary targets that improve learning stability. Periodically copying weights from the main network synchronizes the target network.

## Q13: What are policy gradient methods?

**A13:** Policy gradient methods directly optimize the policy π(a|s;θ) parameterized by θ, rather than learning value functions. They use gradient ascent on expected return: ∇θ J(θ) = E[∇θ log π(a|s;θ) Q(s,a)]. The policy gradient theorem shows how to estimate gradients from experience. These methods can handle continuous action spaces and stochastic policies naturally. They're on-policy, requiring fresh samples under the current policy for each update.

## Q14: Explain the REINFORCE algorithm.

**A14:** REINFORCE is a Monte Carlo policy gradient algorithm that estimates gradients using complete episode returns. It updates policy parameters: θ ← θ + α ∇θ log π(a|s;θ) G_t, where G_t is the return from time t. The algorithm requires complete episodes before updating, making it high-variance but unbiased. Variance reduction techniques include baseline subtraction (using value function estimates) and advantage estimation. REINFORCE is simple but sample-inefficient compared to actor-critic methods.

## Q15: What are actor-critic methods?

**A15:** Actor-critic methods combine policy gradients (actor) with value function estimation (critic). The actor updates the policy using policy gradients, while the critic estimates value functions to reduce variance. The critic provides advantages A(s,a) = Q(s,a) - V(s) instead of raw returns. This reduces variance compared to REINFORCE while maintaining bias. Actor-critic methods can update online after each step, improving sample efficiency. They balance benefits of value-based and policy-based methods.

## Q16: Explain advantage functions and their role in A2C/A3C.

**A16:** The advantage function A(s,a) = Q(s,a) - V(s) measures how much better action a is compared to average in state s. Positive advantages indicate better-than-average actions. A2C (Advantage Actor-Critic) uses n-step returns to estimate advantages, updating both actor and critic networks. A3C (Asynchronous Advantage Actor-Critic) uses multiple parallel actors collecting experiences asynchronously, updating a shared network. This parallelization speeds up learning and decorrelates experiences naturally.

## Q17: What is Proximal Policy Optimization (PPO)?

**A17:** PPO is a policy gradient method that prevents large policy updates by clipping the objective function. It maximizes: L(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)], where r(θ) is the probability ratio between new and old policies. The clipping prevents the policy from changing too drastically, improving stability. PPO can perform multiple updates on the same batch of data, improving sample efficiency. It's simpler than TRPO while maintaining similar performance and is widely used in practice.

## Q18: What is reward shaping and when is it useful?

**A18:** Reward shaping modifies the reward function to provide intermediate feedback and guide learning. It adds shaping rewards F(s,a,s') to the original reward, ideally preserving optimal policies. Reward shaping helps in sparse reward environments where the agent rarely receives feedback. However, poor shaping can lead to suboptimal policies or reward hacking. Potential-based reward shaping guarantees policy invariance. Shaping requires domain knowledge and careful design.

## Q19: Explain the discount factor gamma and its impact.

**A19:** The discount factor γ ∈ [0,1] determines how much future rewards are valued relative to immediate rewards. γ close to 1 emphasizes long-term rewards, while γ close to 0 focuses on immediate rewards. It ensures finite returns in infinite-horizon problems. The discount factor affects exploration: higher γ encourages long-term planning, while lower γ promotes myopic behavior. Choosing γ involves trade-offs between computational tractability and planning horizon. In practice, γ is often set between 0.9 and 0.99.

## Q20: Compare Monte Carlo and temporal difference methods.

**A20:** Monte Carlo methods learn from complete episodes, using actual returns G_t as targets. They're unbiased but high-variance and require episodic tasks. Temporal difference methods bootstrap, using estimates of future values as targets. They're biased but lower-variance and can learn online from incomplete sequences. TD methods are more sample-efficient and applicable to continuing tasks. MC methods are simpler but less practical. TD(λ) provides a spectrum between TD(0) and MC methods using eligibility traces.
