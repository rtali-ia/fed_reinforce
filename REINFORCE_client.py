
import numpy as np
import gym
import torch

def calculate_returns(rewards, dones, gamma):
    result = np.empty_like(rewards)
    result[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        result[t] = rewards[t] + gamma * (1 - dones[t]) * result[t + 1]
    return result

def REINFORCE_client(env, agent, device, gamma=0.99, T=1000):
    # Initialize arrays for storing states, actions, rewards, and done flags
    states = np.empty((T, env.num_envs, agent.N))
    actions = np.empty((T, env.num_envs, agent.M)) if not isinstance(env.action_space, gym.spaces.Discrete) else np.empty((T, env.num_envs))
    rewards = np.empty((T, env.num_envs))
    dones = np.empty((T, env.num_envs))
    
    # Reset the environment to get the initial state
    s_t = env.reset() #This is in CPU

    # Iterate over the time steps
    for t in range(T):
        # Agent selects an action based on the current state
        a_t = agent.act(s_t).cpu().numpy() #This is in CPU

        # Environment advances to the next state and returns reward and done flag
        s_t_next, r_t, d_t = env.step(a_t)

        # Store the state, action, reward, and done flag
        states[t] = s_t
        actions[t] = a_t
        rewards[t] = r_t
        dones[t] = d_t

        # Update the current state
        s_t = s_t_next

    # Compute returns from the rewards
    returns = calculate_returns(rewards, dones, gamma)

    # Update the agent's policy based on the collected experience
    agent.learn(states, actions, returns, device)
    
    #List down the model parameters
    model_params = list(agent.p.parameters())

    # Send to CPU
    cpu_params = [ param.cpu().detach() for param in model_params]
        
    # Return the updated policy parameters and the average reward
    return cpu_params, rewards.sum() / dones.sum()

