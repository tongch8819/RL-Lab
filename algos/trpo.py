from cral_azsc.config_default import cwd
from cral_azsc.utils import build_logger

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

import math
import numpy as np
import scipy.optimize
from collections import namedtuple
import matplotlib.pyplot as plt

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')


class TRPO:
    """
    Trust Region Policy Optimization
    """
    def __init__(self, env,
        model_path = cwd + '/model/policy/trpo_ac.hdf5',
        fig_path = cwd + '/fig/training/trpo.jpg'
    ):

        self._bind(env)
        self.model_path = model_path
        self.fig_path = fig_path
        self.algo_name = "TRPO"

        self.avg_ret_hist = []
        self.pi_loss = []
        self.v_loss = []

    def _bind(self, env):
        self.env = env

    def train(self, epochs, seed, log_paths):
        # seed = 543
        torch.manual_seed(seed)
        np.random.seed(seed)

        gamma = 0.995
        tau = 0.97
        l2_reg = 1e-3
        max_kl = 1e-2
        damping = 1e-1
        log_interval = 1

        act_low, act_high = 0.0, 1.0

        # sum steps of trajectories
        batch_size = 5000
        max_ep_len = 1000

        env = self.env

        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        env.seed(seed)

        policy_net = Policy(num_inputs, num_actions)
        value_net = Value(num_inputs)

        rew_logger = build_logger(self.algo_name + '_rew', log_paths[0])
        rew_logger.info("Epoch,Length,Reward")
        loss_logger = build_logger(self.algo_name + '_loss', log_paths[1])
        loss_logger.info("Lagrange_multiplier,Loss_norm")

        def select_action(state):
            state = torch.from_numpy(state).unsqueeze(0)
            action = policy_net.act(state)
            action = np.clip(action, act_low, act_high)
            return action

        def update_params(batch):
            rewards = torch.Tensor(batch.reward)
            masks = torch.Tensor(batch.mask)
            actions = torch.Tensor(np.concatenate(batch.action, 0))
            states = torch.Tensor(np.array(batch.state))
            values = value_net(Variable(states))

            returns = torch.Tensor(actions.size(0),1)
            deltas = torch.Tensor(actions.size(0),1)
            advantages = torch.Tensor(actions.size(0),1)

            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(rewards.size(0))):
                returns[i] = rewards[i] + gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
                advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

                prev_return = returns[i, 0]
                prev_value = values.data[i, 0]
                prev_advantage = advantages[i, 0]

            targets = Variable(returns)

            # Original code uses the same LBFGS to optimize the value loss
            def get_value_loss(flat_params):
                set_flat_params_to(value_net, torch.Tensor(flat_params))
                for param in value_net.parameters():
                    if param.grad is not None:
                        param.grad.data.fill_(0)

                values_ = value_net(Variable(states))

                value_loss = (values_ - targets).pow(2).mean()

                # weight decay
                for param in value_net.parameters():
                    value_loss += param.pow(2).sum() * l2_reg
                value_loss.backward()
                return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

            flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
                get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
            set_flat_params_to(value_net, torch.Tensor(flat_params))

            advantages = (advantages - advantages.mean()) / advantages.std()

            action_means, action_log_stds, action_stds = policy_net(Variable(states))
            fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

            def get_loss(volatile=False):
                if volatile:
                    with torch.no_grad():
                        action_means, action_log_stds, action_stds = policy_net(Variable(states))
                else:
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
                        
                log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
                return action_loss.mean()


            def get_kl():
                mean1, log_std1, std1 = policy_net(Variable(states))

                mean0 = Variable(mean1.data)
                log_std0 = Variable(log_std1.data)
                std0 = Variable(std1.data)
                kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                return kl.sum(1, keepdim=True)

            trpo_step(policy_net, get_loss, get_kl, max_kl, damping, loss_logger)

        # running_state = ZFilter((num_inputs,), clip=5)
        # running_reward = ZFilter((1,), demean=False, clip=10)

        # Main loop
        for epoch in range(epochs):
            memory = Memory()

            num_steps = 0
            reward_batch = 0
            tau_cnt = 0
            while num_steps < batch_size:
                state = env.reset()
                ep_len, ep_ret = 0, 0
                # state = running_state(state)

                ep_ret = 0
                for t in range(max_ep_len): # Don't infinite loop while learning
                    action = select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    ep_ret += reward

                    # next_state = running_state(next_state)

                    mask = 1
                    if done:
                        mask = 0

                    memory.push(state, np.array([action]), mask, next_state, reward)

                    # if render:
                        # env.render()
                    if done:
                        break

                    state = next_state
                # num_steps += (t-1)
                ep_len = t + 1
                num_steps += ep_len
                tau_cnt += 1
                # reward_batch += reward_sum
                print(f"Epoch k: {epoch+1} Trajectory tau: {tau_cnt} Episode T: {ep_len} Reward: {ep_ret:.3f}")
                rew_logger.info(f"{epoch+1},{ep_len},{ep_ret}")
                

            # reward_batch /= tau_cnt 
            batch = memory.sample()
            update_params(batch)

            # if i_episode % log_interval == 0:
                # print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                    # i_episode, reward_sum, reward_batch))
        # End of main loop
        self.plot_summary()
        self.save_model(policy_net)

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


# utils
def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    # print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("fval after", newfval.item())
            return True, xnew
    return False, x


def trpo_step(model, get_loss, get_kl, max_kl, damping, logger):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
    logger.info(f"{lm[0]},{loss_grad.norm()}")

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        if len(action_mean.shape) == len(self.action_log_std.shape) - 1:
            action_mean = action_mean.unsqueeze(dim=0)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def act(self, x):
        action_mean, _, action_std = self.forward(Variable(x))
        action = torch.normal(action_mean, action_std)
        action = action.data[0].numpy()
        return action



class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)




# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape
