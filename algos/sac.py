from cral_azsc.config_default import cwd
from cral_azsc.utils import combined_shape, mlp, build_logger
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import matplotlib.pyplot as plt


class SAC:
    """
    Soft Actor Critic
    """

    def __init__(self, env,
        model_path=cwd + '/model/policy/sac_ac.hdf5',
        fig_path=cwd + '/fig/training/sac.jpg',
    ):
        self._bind(env)
        self.model_path = model_path
        self.fig_path = fig_path
        self.algo_name = 'SAC'

        self.avg_ret_hist = []
        self.pi_loss = []
        self.q_loss = []

    def _bind(self, env):
        self.env = env
        self.test_env = deepcopy(env)

    def train(self, epochs, seed, log_paths):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # coefficient in targets computation
        alpha=0.2 
        gamma=0.99 
        polyak=0.995 

        lr=1e-3 

        start_steps=1000 
        update_after=1000 
        update_every=100 + 1
        batch_size=100 

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = env.action_space.high[0]
        act_low, act_high = 0.0, 1.0

        num_test_episodes=5
        max_ep_len=1000 

        # epochs=100 
        steps_per_epoch=4000 
        total_steps = steps_per_epoch * epochs
        if start_steps >= total_steps:
            print("warning: exploitation start idx is so large that training ignores it")

        # logger_kwargs=dict() 
        # save_freq=1
    
        # logger = EpochLogger(**logger_kwargs)
        # logger.save_config(locals())
        # rew_logger = build_logger(self.algo_name + '_rew', cwd + '/log/' + self.algo_name + '_rew.log')
        rew_logger = build_logger(self.algo_name + '_rew', log_paths[0])
        rew_logger.info("Epoch,Length,Reward")
        loss_logger = build_logger(self.algo_name + '_loss', log_paths[1])
        loss_logger.info("Loss_pi,Loss_q")

        env, test_env = self.env, self.test_env
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]

        # Create actor-critic module and target networks
        actor_critic=MLPActorCritic 
        ac_kwargs=dict()
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        ac_targ = deepcopy(ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False
            
        # Set up optimizers for policy and q-function
        pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
        q_optimizer = Adam(q_params, lr=lr)

        # Experience buffer
        replay_size=int(1e6) 
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
        # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

        # Set up function for computing SAC Q-losses
        def compute_loss_q(data):
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

            q1 = ac.q1(o,a)
            q2 = ac.q2(o,a)

            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                a2, logp_a2 = ac.pi(o2)

                # Target Q-values
                q1_pi_targ = ac_targ.q1(o2, a2)
                q2_pi_targ = ac_targ.q2(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup)**2).mean()
            loss_q2 = ((q2 - backup)**2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            q_info = dict(Q1Vals=q1.detach().numpy(),
                        Q2Vals=q2.detach().numpy())

            return loss_q, q_info

        # Set up function for computing SAC pi loss
        def compute_loss_pi(data):
            o = data['obs']
            pi, logp_pi = ac.pi(o)
            q1_pi = ac.q1(o, pi)
            q2_pi = ac.q2(o, pi)
            q_pi = torch.min(q1_pi, q2_pi)

            # Entropy-regularized policy loss
            loss_pi = (alpha * logp_pi - q_pi).mean()

            # Useful info for logging
            pi_info = dict(LogPi=logp_pi.detach().numpy())

            return loss_pi, pi_info

        # Set up model saving
        # logger.setup_pytorch_saver(ac)

        def update(data):
            # First run one gradient descent step for Q1 and Q2
            q_optimizer.zero_grad()
            loss_q, q_info = compute_loss_q(data)
            loss_q.backward()
            q_optimizer.step()

            # Record things
            # logger.store(LossQ=loss_q.item(), **q_info)
            self.q_loss.append(loss_q.item())

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)

            self.pi_loss.append(loss_pi.item())

            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            # logger.store(LossPi=loss_pi.item(), **pi_info)

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
            return loss_pi, loss_q

        def get_action(o, deterministic=False):
            a = ac.act(torch.as_tensor(o, dtype=torch.float32), 
                        deterministic)
            # return np.clip(a, act_low, act_high)
            return a

        def test_agent(epoch):
            for j in range(num_test_episodes):
                o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
                while not(d or (ep_len == max_ep_len)):
                    # Take deterministic actions at test time 
                    o, r, d, _ = test_env.step(get_action(o, True))
                    ep_ret += r
                    ep_len += 1
                # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
                print(f"Epoch k: {epoch} Episode Num: {j+1} Episode T: {ep_len} Reward: {ep_ret:.3f}")
                self.avg_ret_hist.append( ep_ret / ep_len )

        # Prepare for interaction with environment
        # start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            
            epoch = t // steps_per_epoch + 1
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > start_steps:
                a = get_action(o)
            else:
                # a = env.action_space.sample()
                obs_obj = self.env.s_codec.decode_numpy(o)
                a = self.env.action_space.sample(obs_obj)

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                print(f"Epoch k: {epoch} Episode Num: train Episode T: {ep_len} Reward: {ep_ret:.3f}")
                rew_logger.info(f"{epoch},{ep_len},{ep_ret:.3f}")
                self.avg_ret_hist.append(
                    ep_ret / ep_len
                )
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    loss_pi, loss_q = update(data=batch)
                print("Loss pi: {:.3f}   Loss Q(s,a): {}".format(
                    loss_pi.item(), loss_q.item()))
                loss_logger.info(f"{loss_pi.item()},{loss_q.item()}")


            # End of epoch handling
            if (t+1) % steps_per_epoch == 0:

                # Save model
                # if (epoch % save_freq == 0) or (epoch == epochs):
                    # logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                test_agent(epoch)

                # Log info about epoch
                # logger.log_tabular('Epoch', epoch)
                # logger.log_tabular('EpRet', with_min_and_max=True)
                # logger.log_tabular('TestEpRet', with_min_and_max=True)
                # logger.log_tabular('EpLen', average_only=True)
                # logger.log_tabular('TestEpLen', average_only=True)
                # logger.log_tabular('TotalEnvInteracts', t)
                # logger.log_tabular('Q1Vals', with_min_and_max=True)
                # logger.log_tabular('Q2Vals', with_min_and_max=True)
                # logger.log_tabular('LogPi', with_min_and_max=True)
                # logger.log_tabular('LossPi', average_only=True)
                # logger.log_tabular('LossQ', average_only=True)
                # logger.log_tabular('Time', time.time()-start_time)
                # logger.dump_tabular()
        # End of main loop
        # self.plot_avg_ret()
        self.plot_summary()
        self.save_model(ac)


    def plot_avg_ret(self, ax=None):
        x = range(len(self.avg_ret_hist))
        if x == 0:
            print("Nothing to plot. Skip")
            return
        y = self.avg_ret_hist

        title = "{} average return".format(self.algo_name)
        if ax is None:
            plt.plot(x, y)
            plt.title(title)
            plt.savefig(self.fig_path)
            print("\nPlot saved at: " + self.fig_path)
        else:
            ax.plot(x, y)
            ax.set_title(title)

    def save_model(self, model):
        torch.save(model, self.model_path)
        print("Model is saved at", self.model_path)

    def plot_pi_loss(self, ax):
        x = range(len(self.pi_loss))
        y = self.pi_loss
        ax.plot(x, y)
        ax.set_title("Pi(s) Loss")

    def plot_q_loss(self, ax):
        x = range(len(self.q_loss))
        y = self.q_loss
        ax.plot(x, y)
        ax.set_title("Q(s,a) Loss")

    def plot_summary(self):
        width, height = 10, 10
        fig, axes = plt.subplots(2, 2, figsize=(width, height))
        fig.suptitle("Params: None")

        self.plot_avg_ret(ax=axes[0][0])
        self.plot_pi_loss(ax=axes[0][1])
        self.plot_q_loss(ax=axes[1][1])

        plt.savefig(self.fig_path)
        print("\nPlot saved at: " + self.fig_path)




LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # pi_action = torch.tanh(pi_action)
        pi_action = torch.sigmoid(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}