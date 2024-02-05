
import torch

class Policy:
    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        (torch.distributions.Distribution)

        s_t (np.ndarray): the current state
        '''
        raise NotImplementedError

    def act(self, s_t):
        '''
        s_t (np.ndarray): the current state
        Because of environment vectorization, this will produce
        E actions where E is the number of parallel environments.
        '''
        
        a_t = self.pi(s_t).sample()
        return a_t

    def learn(self, states, actions, returns, device):
        '''
        states (np.ndarray): the list of states encountered during
                             rollout
        actions (np.ndarray): the list of actions encountered during
                              rollout
        returns (np.ndarray): the list of returns
        '''
        actions = torch.tensor(actions).to(device)
        returns = torch.tensor(returns).to(device)
        states = torch.tensor(states).to(device)

        log_prob = self.pi(states).log_prob(actions)
        loss = torch.mean(-log_prob*returns)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class DiagonalGaussianPolicy(Policy):
    def __init__(self, env, device, lr=1e-2):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        '''
        self.device = device
        self.N = env.observation_space.shape[0]
        self.M = env.action_space.shape[0]
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(self.N, self.M),
        ).double()

        self.log_sigma = torch.ones(self.M, dtype=torch.double, requires_grad=True).to(self.device)

        self.opt = torch.optim.Adam(list(self.mu.parameters()) + [self.log_sigma], lr=lr)

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state
        '''
        s_t = torch.as_tensor(s_t).double().to(self.device)
        mu = self.mu(s_t)
        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        return pi

class CategoricalPolicy(Policy):
    def __init__(self, env, device, lr=1e-2):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        '''
        self.device = device
        self.N = env.observation_space.shape[0]
        self.M = env.action_space.n
        self.p = torch.nn.Sequential(
            torch.nn.Linear(self.N, self.M),
        ).double().to(self.device) #In GPU

        self.opt = torch.optim.Adam(self.p.parameters(), lr=lr) #In GPU

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state
        '''
        s_t = torch.as_tensor(s_t).double().to(self.device) #In GPU
        p = self.p(s_t) #In GPU
        pi = torch.distributions.Categorical(logits=p)  #In GPU
        return pi
