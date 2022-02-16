from cral_azsc.config_default import cwd
from cral_azsc.utils import discount_cumsum, combined_shape, mlp, count_vars 
import numpy as np

import torch
import torch.nn as nn


from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    """

    def __init__(self, env,
        model_path=cwd + '/model/policy/td3_ac.hdf5',
        fig_path=cwd + '/fig/training/td3.jpg',
    ):
        self._bind(env)
        self.model_path = model_path
        self.fig_path = fig_path
        self.algo_name = 'TD3'

        self.avg_ret_hist = []
        self.pi_loss = []
        self.q_loss = []

    def _bind(self, env):
        self.env = env
        self.test_env = deepcopy(env)

    def train(self, epochs):
        # seed (int): Seed for random number generators.
        seed=0 
        torch.manual_seed(seed)
        np.random.seed(seed)

        # gamma (float): Discount factor. (Always between 0 and 1.)
        gamma=0.9 
        # polyak (float): Interpolation factor in polyak averaging for target 
        #     networks. Target networks are updated towards main networks 
        #     according to:

        #     .. math:: \\theta_{\\text{targ}} \\leftarrow 
        #         \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

        #     where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
        #     close to 1.)
        polyak=0.9

        # pi_lr (float): Learning rate for policy.
        pi_lr=1e-3 
        # q_lr (float): Learning rate for Q-networks.
        q_lr=1e-3 

        # start_steps (int): Number of steps for uniform-random action selection,
        #     before running real policy. Helps exploration.
        start_steps=1000 
        # update_after (int): Number of env interactions to collect before
        #     starting to do gradient descent updates. Ensures replay buffer
        #     is full enough for useful updates.
        update_after=1000 
        # update_every (int): Number of env interactions that should elapse
        #     between gradient descent updates. Note: Regardless of how long 
        #     you wait between updates, the ratio of env steps to gradient steps 
        #     is locked to 1. 
        # - set it to odd numbers to make tail print work
        update_every=100 + 1
        # batch_size (int): Minibatch size for SGD.
        batch_size=100 

        # act_noise (float): Stddev for Gaussian exploration noise added to 
        #     policy at training time. (At test time, no noise is added.)
        # act_noise=0.1 
        noise_mu, noise_sigma = 0, 1
        # target_noise (float): Stddev for smoothing noise added to target 
        #     policy. \sigma
        target_noise=0.2 
        # noise_clip (float): Limit for absolute value of target policy 
        #     smoothing noise. small c
        noise_clip=0.5 
        # policy_delay (int): Policy will only be updated once every 
        #     policy_delay times for each update of the Q-networks.
        policy_delay=2 
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = env.action_space.high[0]
        act_low, act_high = 0.0, 1.0


        # num_test_episodes (int): Number of episodes to test the deterministic
        #     policy at the end of each epoch.
        num_test_episodes=5
        # max_ep_len (int): Maximum length of trajectory / episode / rollout.
        max_ep_len=1000 

        # epochs (int): Number of epochs to run and train agent.
        # epochs=100 
        # steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
        #     for the agent and the environment in each epoch.
        steps_per_epoch=1000 
        total_steps = steps_per_epoch * epochs
        if start_steps >= total_steps:
            print("warning: exploitation start idx is so large that training ignores it")

        # logger_kwargs (dict): Keyword args for EpochLogger.
        # logger_kwargs=dict() 
        # save_freq (int): How often (in terms of gap between epochs) to save
        #     the current policy and value function.
        # save_freq=1

        # logger = EpochLogger(**logger_kwargs)
        # logger.save_config(locals())


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
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
        q_optimizer = Adam(q_params, lr=q_lr)

        # replay_size (int): Maximum length of replay buffer.
        replay_size=int(1e6) 
        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
        # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

        # Set up function for computing TD3 Q-losses
        def compute_loss_q(data):
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

            q1 = ac.q1(o,a)
            q2 = ac.q2(o,a)

            # Bellman backup for Q functions
            with torch.no_grad():
                pi_targ = ac_targ.pi(o2)

                # Target policy smoothing
                epsilon = torch.randn_like(pi_targ) * target_noise
                epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
                a2 = pi_targ + epsilon
                # a2 = torch.clamp(a2, -act_limit, act_limit)
                a2 = torch.clamp(a2, act_low, act_high)

                # Target Q-values
                q1_pi_targ = ac_targ.q1(o2, a2)
                q2_pi_targ = ac_targ.q2(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + gamma * (1 - d) * q_pi_targ

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup)**2).mean()
            loss_q2 = ((q2 - backup)**2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            loss_info = dict(Q1Vals=q1.detach().numpy(),
                            Q2Vals=q2.detach().numpy())

            return loss_q, loss_info

        # Set up function for computing TD3 pi loss
        def compute_loss_pi(data):
            o = data['obs']
            q1_pi = ac.q1(o, ac.pi(o))
            return -q1_pi.mean()


        # Set up model saving
        # logger.setup_pytorch_saver(ac)

        def update(data, timer):
            # First run one gradient descent step for Q1 and Q2
            q_optimizer.zero_grad()
            loss_q, loss_info = compute_loss_q(data)
            loss_q.backward()
            q_optimizer.step()

            # Record things
            # logger.store(LossQ=loss_q.item(), **loss_info)
            self.q_loss.append(loss_q.item())

            # Possibly update pi and target networks
            loss_pi = None
            if timer % policy_delay == 0:

                # Freeze Q-networks so you don't waste computational effort 
                # computing gradients for them during the policy learning step.
                for p in q_params:
                    p.requires_grad = False

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi = compute_loss_pi(data)
                loss_pi.backward()
                pi_optimizer.step()

                # Unfreeze Q-networks so you can optimize it at next DDPG step.
                for p in q_params:
                    p.requires_grad = True

                # Record things
                # logger.store(LossPi=loss_pi.item())
                self.pi_loss.append(loss_pi.item())

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
            return loss_pi, loss_q

        def get_action(o, noise_scale):
            a = ac.act(torch.as_tensor(o, dtype=torch.float32))
            a += noise_scale * np.random.randn(act_dim)
            # return np.clip(a, -act_limit, act_limit)
            return np.clip(a, act_low, act_high)

        def test_agent(epoch):
            for j in range(num_test_episodes):
                o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
                while not(d or (ep_len == max_ep_len)):
                    # Take deterministic actions at test time (noise_scale=0)
                    o, r, d, _ = test_env.step(get_action(o, 0))
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
            # use the learned policy (with some noise, via act_noise). 
            if t > start_steps:
                act_noise = np.random.normal(noise_mu, noise_sigma, 1).item()
                a = get_action(o, act_noise)
            else:
                # a = env.action_space.sample()
                a = self.env.action_space.sample(self.env.s_codec.decode(o, method='numpy'))

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
                self.avg_ret_hist.append(
                    ep_ret / ep_len
                )
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    loss_pi, loss_q = update(data=batch, timer=j)
                if loss_pi is None:
                    print("Loss pi: None   Loss Q(s,a): {}".format(loss_q.item()))
                else:
                    print("Loss pi: {:.3f}   Loss Q(s,a): {}".format(
                        loss_pi.item(), loss_q.item()))


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
                # logger.log_tabular('LossPi', average_only=True)
                # logger.log_tabular('LossQ', average_only=True)
                # logger.log_tabular('Time', time.time()-start_time)
                # logger.dump_tabular()
        # End of main loop
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


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
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
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
