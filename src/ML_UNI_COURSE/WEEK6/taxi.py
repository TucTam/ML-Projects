import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size,action_size))
qtable, qtable.shape

# Policy function
def greedy_policy(qtable, state, explo_rate):
    if np.random.uniform(0,1) < explo_rate:
        return np.random.randint(0,6)
    else:
        return np.argmax(qtable[state])
    
# Evaluates/tests the policy over a single episode
def policy_eval(qtable, n_steps, discount, explo_rate):
    state = env.reset()[0]
    total_steps = 0
    total_reward = 0
    for step in range(n_steps):
        action = greedy_policy(qtable, state, explo_rate)
        new_state, reward, done, truncated, info = env.step(action)
        state = new_state
        total_reward += reward*discount**step
        
        if done: 
            total_steps = step + 1  
            break
        else:
            total_steps = step + 1

    return total_reward, total_steps    
                
# Evaluates/tests the policy fully over many iterations
def full_policy_eval(qtable, n_episodes, n_steps, discount, explo_rate):
    episode_returns = []
    episode_steps = []
    for episode in range(n_episodes):
        total_reward, steps = policy_eval(qtable, n_steps, discount, explo_rate)
        episode_steps.append(steps)
        episode_returns.append(total_reward)
        
    return np.mean(episode_returns), np.min(episode_returns), np.max(episode_returns), np.std(episode_returns), np.mean(episode_steps)

# Function for training policy
def training(Q, lr, discount, min_explorate, explo_rate, decay, n_episodes, n_steps):
    hist = []
    for episode in range(n_episodes):
        state = env.reset()[0]
        
        for step in range(n_steps):
            # Let the agent act
            action = greedy_policy(qtable, state, explo_rate)
            new_state, reward, done, truncated, info = env.step(action)        
            
            # Updating Qtable (policy) with bellmans equation
            Q[state,action] = Q[state, action] + lr*(reward+discount*np.max(Q[new_state])-Q[state,action])
            state = new_state
            if done: break
        
        exploration_rate = max(min_explorate, explo_rate * decay)
        if episode % 10 == 0 or episode == 1:
            mean_return, smallest_return, best_return, std_return, mean_steps = full_policy_eval(Q, 10, n_steps, discount, 0)
            hist.append([episode, mean_return,smallest_return,best_return,std_return, mean_steps])
    return Q, hist

# Hyperparameters
learning_rate = 0.1
discount = 0.995
exploration_rate = 1
min_explorate = 0.1
decay = 0.995
n_episodes = 5000
n_steps = 100

qtable, hist_eval = training(qtable, learning_rate, discount, min_explorate, exploration_rate, decay, n_episodes, n_steps)

# Plotting the evaluation history
hist_eval = np.array(hist_eval)
plt.plot(hist_eval[:,0], hist_eval[:, 1])
plt.title("Average return over 10 episodes for every 10 episodes of training")
plt.xlabel("Training Episodes")
plt.ylabel("average return over 10 episodes")

# Testing the model
# Test hyperparams
test_episodes = 10
test_steps = n_steps
test_discount = discount
test_exploration = 0
mean_return, smallest_return, best_return, std_return, mean_steps = full_policy_eval(qtable, test_episodes, test_steps, test_discount, test_exploration)

print(f"Average return over {test_episodes} episodes is: {mean_return}")
print(f"Smallest return of the {test_episodes} episodes is: {smallest_return}")
print(f"Greatest return of the {test_episodes} episodes is: {best_return}")
print(f"std return of the {test_episodes} episodes is: {std_return}")
print(f"Average amounts of steps over {test_episodes} episodes is: {mean_steps}")

plt.show()
