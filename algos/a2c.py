from cral_azsc.action import Action
from cral_azsc.baseline_lab import CloudEnvPrimitive
from cral_azsc.config_default import req_config_default, cwd

import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt

import argparse

args_mapping = {
    'agg' : 'aggregate',
    'util' : 'utilization_rate',
    'ava' : 'availability_difference',
    'fair' : 'fairness_variance',
    'cred' : 'credit_fairness',
}

def t(x): return torch.from_numpy(x).float()

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x):
        return self.model(x)

def train(reward_method_alias, epis=500):
    """
    - reward_method_alias: alias for cmdlet args 
    """
    n_users = 2
    n_quota = 1000
    reward_method = args_mapping.get(reward_method_alias, 'aggregate')
    # reward_method='aggregate'
    # reward_method='utilization_rate'
    # reward_method='availability_difference'
    # reward_method='fairness_variance'
    # reward_method='credit_fairness'
    fig_path = cwd + '/fig/a2c_new_reward_shaping/{}.jpg'.format(reward_method)
    env = CloudEnvPrimitive(n_users, n_quota, 100,
        req_bnd=req_config_default['req_bnd'], 
        reward_method=reward_method
    )

    state_dim = n_users * 3 + 1
    # n_actions = 5
    actor, critic = Actor(state_dim, n_users), Critic(state_dim)
    adam_a, adam_c = torch.optim.Adam(actor.parameters(), lr=0.01), \
        torch.optim.Adam(critic.parameters(), lr=0.01)
    gamma = 0.99

    episode_rewards = []
    for i in range(epis):
        done = False
        total_reward = 0
        state = env.reset()

        print("Episode {} running...".format(i))
        for j in range(100):
            state_arr = state.encode('naive')
        
            action_arr = actor(t(state_arr))
            action = Action(action_arr.detach().data.numpy())

            next_state, reward, done, info = env.step(action)
            advantage = reward - critic(t(state_arr))
            if not done:
                advantage += gamma * critic(t(state_arr))
            
            total_reward += reward
            state = next_state

            critic_loss = advantage.pow(2).mean()
            adam_c.zero_grad()
            critic_loss.backward()
            adam_c.step()

            # actor_loss = - dist.log_prob(action) * advantage.detach()
            # print(action_arr)
            actor_loss = action_arr.max() * advantage.detach()
            adam_a.zero_grad()
            actor_loss.backward()
            adam_a.step()
        
        episode_rewards.append( total_reward )

    return episode_rewards, fig_path

def plot_episode_rewards(fig_path, episode_rewards):
    x = np.arange(len(episode_rewards))
    plt.scatter(x, episode_rewards, s=2)
    plt.title("Episode rewards")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig(fig_path)
    print("\n\nDone. Please check figure at: {}".format(fig_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reward_type", type=str, help="agg, util, fair, cred")
    parser.add_argument("epi_len", type=int, help="episodic length")
    args = parser.parse_args()


    episode_rewards, fig_path = train(args.reward_type, args.epi_len)
    plot_episode_rewards(fig_path, episode_rewards)

if __name__ == "__main__":
    main()