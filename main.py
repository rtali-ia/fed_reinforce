import gym
import numpy as np
import torch
import random
import seaborn as sns
from environment import VectorizedEnvWrapper
from reinforcement_learning_policies import CategoricalPolicy, DiagonalGaussianPolicy
from REINFORCE_client import REINFORCE_client
from geom_median.torch import compute_geometric_median
import time
import sys

#Connect to GPU if available else run on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(env_name = "CartPole-v1", n_envs = 32, ifPrint = False):
    # Initialize the environment
    env = VectorizedEnvWrapper(gym.make(env_name), num_envs=n_envs)
    N = env.observation_space.shape[0]
    M = env.action_space.n

    # Create the global model - In GPU
    global_model = torch.nn.Sequential(
                torch.nn.Linear(N, M),
            ).double()
    
    global_dict = global_model.state_dict() #In GPU
    
    # Copy the global model to create client models
    client_models = [CategoricalPolicy(env, device, lr=1e-1) for _ in range(10)] #These are in GPU
    
    for model in client_models:
        model.p.load_state_dict(global_model.state_dict()) #In GPU

    # Training loop
    
    epoch_rewards = []
    
    for epoch in range(100):
        gradients, rewards = [], []

        # Collect gradients and rewards from each client
        for client_model in client_models:
            grad, reward = REINFORCE_client(env, client_model, device) #Need to make sure the gradient and reward vectors are in CPU.
            gradients.append(grad) #In CPU
            rewards.append(reward)

        # Attack: Sign-flipping
        malicious_clients = random.sample(range(10), 3)
        for client_idx in malicious_clients:
            for grad_idx in range(len(global_model.state_dict())):
                gradients[client_idx][grad_idx] = -2.5*gradients[client_idx][grad_idx]

        # Compute geometric median of gradients
        median_gradient = compute_geometric_median(gradients, weights=None)

        # Update global model - This is still in GPU
        i=0
        for k in global_dict.keys():
            global_dict[k] = median_gradient.median[i]
            i=i+1
        global_model.load_state_dict(global_dict)


        # Synchronize client models with the global model
        for model in client_models:
            model.p.load_state_dict(global_model.state_dict())

        # Print model parameters for verification
        if ifPrint:
            for key in global_model.state_dict():
                print(key, global_model.state_dict()[key])

        # Store average reward
        epoch_rewards.append(sum(rewards) / len(rewards))

    # Plot reward trends
    fig = sns.lineplot(x=range(len(epoch_rewards)), y=epoch_rewards)
    fig_1 = fig.get_figure()
    fig_1.savefig("out.png")

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print("Training on your primary GPU")
        print("\nAt this moment we can use only a single GPU, the primary GPU for this trainig is: ", torch.cuda.get_device_name(0))
    else:
        print("Training on CPU")
    
    start_time = time.time()
    
    if len(sys.argv) == 4:
        print("\nEnvironment: ", sys.argv[1])
        print("\nNumber of Environments: ", sys.argv[2])
        print("\nIf you want to print the model parameters: ", sys.argv[3])
        
        main(sys.argv[1], int(sys.argv[2]), bool(sys.argv[3]))
        
    else:
        print("\nPlease provide the environment name, number of environments and ifPrint as arguments")
        print("\nExample: python main.py CartPole-v1 32 False")
        print("\nWe are using the default values: CartPole-v1, 32, False")
        main()
        
    print("\n--- Total Run Time for GMFedReinforce = %s seconds ---" % (time.time() - start_time))
    print('\nAll Done!')