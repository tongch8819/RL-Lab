from cral_azsc.config_default import cwd
from cral_azsc.utils import discount_cumsum, combined_shape, mlp, count_vars 

import numpy as np

import torch
import torch.nn as nn

from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

class DDPG:
    def __init__(self, env,
        model_path=cwd + '/model/policy/ddpg_ac.hdf5',
        fig_path=cwd + '/fig/training/ddpg.jpg',
    ):
        self._bind(env)
        self.model_path = model_path
        self.fig_path = fig_path

        self.avg_ret_hist = []
        self.pi_loss = []
        self.q_loss = []

    def _bind(self, env):
        self.env = env
        # Why do we need a test env?
        # When test process is invoked, the training env may not stop at all.
        self.test_env = deepcopy(env)

    def train(self, epochs):
        seed=0 
        torch.manual_seed(seed)
        np.random.seed(seed)

        gamma=0.9     # coefficient when computing targets y
        polyak=0.9   # \pho used in update of targ network

        pi_lr=1e-3   # \pi learning rate
        q_lr=1e-3    # Q learing rate

        start_steps=1000   # exploration stop idx 
        update_after=1000 
        # | ------ update stage ------ |
        # | up_eve_1 | up_eve_2 | ..   |
        update_every=100
        batch_size=100 

        noise_mu, noise_sigma = 0, 1
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = self.env.action_space.high[0]
        act_low, act_high = 0.0, 1.0


        num_test_episodes=5 
        max_ep_len=1000 
        # save_freq=1

        # epochs=100 
        steps_per_epoch=1000  
        total_steps = steps_per_epoch * epochs  # Main loop length
        if start_steps >= total_steps:
            print("warning: exploitation start idx is so large that training ignores it")

        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Create actor-critic module and target networks
        ac_kwargs = dict() 
        actor_critic = MLPActorCritic
        ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        ac_targ = deepcopy(ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and q-function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

        # Experience buffer
        replay_size = int(1e6) 
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q])

        # Set up function for computing DDPG pi loss
        def compute_loss_pi(data):
            o = data['obs']
            # we cannot freeze all grads since grads in ac.pi
            # are needed during backpropogation.
            q_pi = ac.q(o, ac.pi(o))
            return -q_pi.mean()

        # Set up function for computing DDPG Q-loss
        def compute_loss_q(data):
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

            q = ac.q(o,a)

            # Bellman backup for Q function
            with torch.no_grad():
                q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
                backup = r + gamma * (1 - d) * q_pi_targ

            # MSE loss against Bellman backup
            loss_q = ((q - backup)**2).mean()

            # Useful info for logging
            loss_info = dict(
                QVals=q.detach().numpy()
            )

            return loss_q, loss_info

        def update(data):
            # data is a mini-batch sampled from replay buffer
            # First run one gradient descent step for Q.
            q_optimizer.zero_grad()
            loss_q, loss_info = compute_loss_q(data)

            self.q_loss.append(loss_q.item())

            loss_q.backward()
            q_optimizer.step()

            # Freeze Q-network so you don't waste computational effort 
            # computing gradients for it during the policy learning step.
            for p in ac.q.parameters():
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)

            self.pi_loss.append(loss_pi.item())

            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-network so you can optimize it at next DDPG step.
            for p in ac.q.parameters():
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
            return loss_pi, loss_q

        def get_action(o, noise_scale):
            # select action by clip and noise
            # using model ac instead of ac_targ
            a = ac.act(torch.as_tensor(o, dtype=torch.float32))
            a += noise_scale * np.random.randn(act_dim)
            # return np.clip(a, -act_limit, act_limit)
            return np.clip(a, act_low, act_high)

        def test_agent(epoch):
            avg_ep_rets = []
            for j in range(num_test_episodes):
                o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
                while not(d or (ep_len == max_ep_len)):
                    # Take deterministic actions at test time (noise_scale=0)
                    o, r, d, _ = self.test_env.step(get_action(o, 0))
                    ep_ret += r
                    ep_len += 1
                print(f"Epoch k: {epoch} Episode Num: {j+1} Episode T: {ep_len} Reward: {ep_ret:.3f}")
                # avg_ep_rets.append( ep_ret / ep_len )
                self.avg_ret_hist.append( ep_ret / ep_len )

            # only add one value to plot
            # self.avg_ret_hist.append( sum(avg_ep_rets) / num_test_episodes ) 

        # Prepare for interaction with environment
        # start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if t > start_steps:
                # sample epsilon in clip
                act_noise = np.random.normal(noise_mu, noise_sigma, 1).item()
                a = get_action(o, act_noise)
            else:
                # a = self.env.action_space.sample()
                a = self.env.action_space.sample(self.env.s_codec.decode(o, method='numpy'))

            # Step the env
            o2, r, d, _ = self.env.step(a)
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
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    loss_pi, loss_q = update(data=batch)
                print("Loss pi: {:.3f}   Loss Q(s,a): {}".format(
                    loss_pi.item(), loss_q.item()))

            # End of epoch handling
            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch

                # Save model
                # if (epoch % save_freq == 0) or (epoch == epochs):
                #     logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                test_agent(epoch)

                # Log info about epoch
                # logger.log_tabular('Epoch', epoch)
                # logger.log_tabular('EpRet', with_min_and_max=True)
                # logger.log_tabular('TestEpRet', with_min_and_max=True)
                # logger.log_tabular('EpLen', average_only=True)
                # logger.log_tabular('TestEpLen', average_only=True)
                # logger.log_tabular('TotalEnvInteracts', t)
                # logger.log_tabular('QVals', with_min_and_max=True)
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
        title = "DDPG average return"
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
    

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
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


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

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
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        # output of \pi is not like PPO actor a distribution, but a direct 
        # representation of continuous action

        # no grad accelerates the forward pass
        with torch.no_grad():
            return self.pi(obs).numpy()
