from cral_azsc.utils import discount_cumsum, combined_shape, mlp, count_vars 
from cral_azsc.config_default import cwd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import numpy as np
import matplotlib.pyplot as plt


class VPG:
    """
    Vanilla Policy Gradient
    """

    def __init__(self, env,
        model_path=cwd + '/model/policy/vpg_ac.hdf5',
        fig_path=cwd + '/fig/training/vpg.jpg',
    ):
        self._bind(env)

        self.model_path = model_path
        self.fig_path = fig_path
        self.algo_name = "VPG"

        self.avg_ret_hist = []
        self.pi_loss = []
        self.v_loss = []

    def _bind(self, env):
        self.env = env

    def train(self, epochs):
        # Random seed
        seed=0 
        # seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # epochs=50 
        steps_per_epoch=5000 
        max_ep_len=1000

        gamma=0.9
        lam=0.97 

        pi_lr=3e-4
        vf_lr=1e-3 
        train_v_iters=80 

        # logger_kwargs=dict()
        # save_freq=10

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        # Set up logger and save configuration
        # logger = EpochLogger(**logger_kwargs)
        # logger.save_config(locals())


        # Instantiate environment
        env = self.env
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        # Create actor-critic module
        actor_critic=MLPActorCritic 
        ac_kwargs=dict(
            hidden_sizes=(64, 128, 64), 
            # activation=nn.Tanh
            activation=nn.ReLU
        )  
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        # Set up optimizers for policy and value function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

        # Sync params across processes
        # sync_params(ac)

        # Count variables
        # var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
        # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up experience buffer
        # local_steps_per_epoch = int(steps_per_epoch / num_procs())
        local_steps_per_epoch = int(steps_per_epoch)
        buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

        # Set up function for computing VPG policy loss
        def compute_loss_pi(data):
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

            # Policy loss
            pi, logp = ac.pi(obs, act)
            loss_pi = -(logp * adv).mean()

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            pi_info = dict(kl=approx_kl, ent=ent)

            return loss_pi, pi_info

        # Set up function for computing value loss
        def compute_loss_v(data):
            obs, ret = data['obs'], data['ret']
            return ((ac.v(obs) - ret)**2).mean()

        # Set up model saving
        # logger.setup_pytorch_saver(ac)

        def update():
            data = buf.get()

            # Get loss and info values before update
            pi_l_old, pi_info_old = compute_loss_pi(data)
            pi_l_old = pi_l_old.item()
            v_l_old = compute_loss_v(data).item()

            # Train policy with a single step of gradient descent
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            print("Loss pi: {:.3f}   KL: {:.3f}".format(loss_pi.item(), pi_info['kl']))

            self.pi_loss.append(loss_pi.item())

            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

            # Value function learning
            for i in range(train_v_iters):
                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(data)

                self.v_loss.append(loss_v.item())

                loss_v.backward()
                # mpi_avg_grads(ac.v)    # average grads across MPI processes
                vf_optimizer.step()

            # Log changes from update
            kl, ent = pi_info['kl'], pi_info_old['ent']
            # logger.store(LossPi=pi_l_old, LossV=v_l_old,
            #             KL=kl, Entropy=ent,
            #             DeltaLossPi=(loss_pi.item() - pi_l_old),
            #             DeltaLossV=(loss_v.item() - v_l_old))

        # Prepare for interaction with environment
        # start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            tau_cnt = 0
            for t in range(local_steps_per_epoch):
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                # save and log
                buf.store(o, a, r, v, logp)
                # logger.store(VVals=v)
                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        # logger.store(EpRet=ep_ret, EpLen=ep_len)
                        print(f"Epoch Loop k: {epoch+1} Trajectory tau: {tau_cnt} Episode T: {ep_len} Reward: {ep_ret:.3f}")
                        self.avg_ret_hist.append(
                            ep_ret / ep_len
                        )
                    o, ep_ret, ep_len = env.reset(), 0, 0
                    tau_cnt += 1


            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     logger.save_state({'env': env}, None)

            # Perform VPG update!
            update()

            # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            # logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            # logger.log_tabular('DeltaLossPi', average_only=True)
            # logger.log_tabular('DeltaLossV', average_only=True)
            # logger.log_tabular('Entropy', average_only=True)
            # logger.log_tabular('KL', average_only=True)
            # logger.log_tabular('Time', time.time()-start_time)
            # logger.dump_tabular()
        # End of Main loop
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
        ax.set_title("Pi(a|s) Loss")

    def plot_v_loss(self, ax):
        x = range(len(self.v_loss))
        y = self.v_loss
        ax.plot(x, y)
        ax.set_title("V(s) Loss")

    def plot_summary(self):
        width, height = 10, 10
        fig, axes = plt.subplots(2, 2, figsize=(width, height))
        fig.suptitle("Params: None")

        self.plot_avg_ret(ax=axes[0][0])
        self.plot_pi_loss(ax=axes[0][1])
        self.plot_v_loss(ax=axes[1][1])

        plt.savefig(self.fig_path)
        print("\nPlot saved at: " + self.fig_path)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        # if isinstance(action_space, Box):
            # self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        # elif isinstance(action_space, Discrete):
            # self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # policy
        self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            a = a.clip(0.0, 1.0)
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}