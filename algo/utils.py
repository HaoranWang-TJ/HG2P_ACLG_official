import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np
import pandas as pd

import random

from functools import total_ordering, reduce
from scipy.special import softmax
from sklearn.linear_model import LinearRegression
# from algo.higl import var

from itertools import compress

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, maxsize=1e6, batch_size=100, reward_func=None, reward_scale=None):
        self.storage = [[] for _ in range(11)]
        self.maxsize = maxsize
        self.next_idx = 0
        self.batch_size = batch_size
        self.reward_func = reward_func
        self.reward_scale = reward_scale

    # Expects tuples of (x, x', ag, g, u, r, d, x_seq, a_seq, ag_seq)
    def add(self, odict):
        assert list(odict.keys()) == ['state', 'next_state', 'achieved_goal', 'next_achieved_goal',
                                      'goal', 'action', 'reward',
                                      'done', 'state_seq', 'actions_seq', 'achieved_goal_seq']
        data = tuple(odict.values())
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            [array.append(datapoint) for array, datapoint in zip(self.storage, data)]
        else:
            [array.__setitem__(self.next_idx, datapoint) for array, datapoint in zip(self.storage, data)]

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, y, ag, ag_next, g, u, r, d, x_seq, a_seq, ag_seq = [], [], [], [], [], [], [], [], [], [], []

        for i in ind:
            X, Y, AG, AG_NEXT, G, U, R, D, obs_seq, acts, AG_seq = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            ag.append(np.array(AG, copy=False))
            ag_next.append(np.array(AG_NEXT, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

            # For off-policy goal correction
            x_seq.append(np.array(obs_seq, copy=False))
            a_seq.append(np.array(acts, copy=False))
            ag_seq.append(np.array(AG_seq, copy=False))

        return np.array(x), np.array(y), np.array(ag), np.array(ag_next), np.array(g), \
               np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
               x_seq, a_seq, ag_seq

    def save(self, file):
        np.savez_compressed(file, idx=np.array([self.next_idx]), x=self.storage[0],
                            y=self.storage[1], ag=self.storage[2], agnext=self.storage[3],
                            g=self.storage[4], u=self.storage[5], r=self.storage[6],
                            d=self.storage[7], xseq=self.storage[8], aseq=self.storage[9],
                            agseq=self.storage[10])

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data['idx'][0])
            self.storage = [data['x'], data['y'], data['ag'], data['agnext'], data['g'],
                            data['u'], data['r'], data['d'], data['xseq'], data['aseq'],
                            data['agseq']]
            self.storage = [list(l) for l in self.storage]

    def __len__(self):
        return len(self.storage[0])


class TrajectoryBuffer:

    def __init__(self, capacity):
        self._capacity = capacity
        self.reset()

    def reset(self):
        self._num_traj = 0  # number of trajectories
        self._size = 0    # number of game frames
        self.trajectory = []

    def __len__(self):
        return self._num_traj

    def size(self):
        return self._size

    def get_traj_num(self):
        return self._num_traj

    def full(self):
        return self._size >= self._capacity

    def create_new_trajectory(self):
        self.trajectory.append([])
        self._num_traj += 1

    def append(self, s):
        self.trajectory[self._num_traj-1].append(s)
        self._size += 1

    def get_trajectory(self):
        return self.trajectory

    def set_capacity(self, new_capacity):
        assert self._size <= new_capacity
        self._capacity = new_capacity


class NormalNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        action = (action + np.random.normal(0, self.sigma,
            size=action.shape[0])).clip(min_action, max_action)
        return action

class OUNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return (self.X + action).clip(min_action, max_action)


def train_adj_net(a_net, states, adj_mat, optimizer, margin_pos, margin_neg,
                  n_epochs=100, batch_size=64, device='cpu', verbose=False):
    if verbose:
        print('Generating training data...')
    dataset = MetricDataset(states, adj_mat)
    if verbose:
        print('Totally {} training pairs.'.format(len(dataset)))
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False, generator=torch.Generator(device))
    n_batches = len(dataloader)

    loss_func = ContrastiveLoss(margin_pos, margin_neg)

    for i in range(n_epochs):
        epoch_loss = []
        for j, data in enumerate(dataloader):
            x, y, label = data
            x = x.float().to(device)
            y = y.float().to(device)
            label = label.long().to(device)
            x = a_net(x)
            y = a_net(y)
            loss = loss_func(x, y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and (j % 50 == 0 or j == n_batches - 1):
                print('Training metric network: epoch {}/{}, batch {}/{}'.format(i+1, n_epochs, j+1, n_batches))

            epoch_loss.append(loss.item())

        if verbose:
            print('Mean loss: {:.4f}'.format(np.mean(epoch_loss)))


class ContrastiveLoss(nn.Module):

    def __init__(self, margin_pos, margin_neg):
        super().__init__()
        assert margin_pos <= margin_neg
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(self, x, y, label):
        # mutually reachable states correspond to label = 1
        dist = torch.sqrt(torch.pow(x - y, 2).sum(dim=1) + 1e-12)
        loss = (label * (dist - self.margin_pos).clamp(min=0)).mean() + ((1 - label) * (self.margin_neg - dist).clamp(min=0)).mean()
        return loss


class MetricDataset(Data.Dataset):

    def __init__(self, states, adj_mat):
        super().__init__()
        n_samples = adj_mat.shape[0]
        self.x = []
        self.y = []
        self.label = []
        for i in range(n_samples - 1):
            for j in range(i + 1, n_samples):
                self.x.append(states[i])
                self.y.append(states[j])
                self.label.append(adj_mat[i, j])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.label = np.array(self.label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.label[idx]


def get_reward_function(env, env_name, absolute_goal=False, binary_reward=False):
    # distance_threshold = env.distance_threshold
    if env_name in ["AntMaze-v1", "PointMaze-v1"]:
        distance_threshold = 2.5
    elif env_name == "AntMazeW-v2":
        distance_threshold = 1
    if absolute_goal and binary_reward:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -float(np.linalg.norm(subgoal - next_ag, axis=-1) > distance_threshold) * scale
            return reward
    elif absolute_goal:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -np.linalg.norm(subgoal - next_ag, axis=-1) * scale
            return reward
    elif binary_reward:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -float(np.linalg.norm(ag + subgoal - next_ag, axis=-1) > distance_threshold) * scale
            return reward
    else:
        def controller_reward(ag, subgoal, next_ag, scale, action=None):
            reward = -np.linalg.norm(ag + subgoal - next_ag, axis=-1) * scale
            return reward

    return controller_reward


def get_mbrl_fetch_reward_function(env, env_name, binary_reward, absolute_goal):
    action_penalty_coeff = 0.0001
    distance_threshold = 0.25
    if env_name in ["FetchPickAndPlace-v1", "FetchPush-v1"]:
        action_penalty_coeff = 0.1
        distance_threshold = 0.05
    if env_name in ["Reacher3D-v0", "Pusher-v0", "FetchPickAndPlace-v1", "FetchPush-v1"]:
        if absolute_goal and not binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward = -np.sum(np.square(ag - subgoal))
                reward -= action_penalty_coeff * np.square(action).sum()
                return reward * scale

        elif absolute_goal and binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward_ctrl = action_penalty_coeff * -np.square(action).sum()
                fail = True
                if np.sqrt(np.sum(np.square(ag - subgoal))) <= distance_threshold:
                    fail = False
                reward = reward_ctrl - float(fail)
                return reward * scale

        elif not absolute_goal and not binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward = -np.sum(np.square(ag + subgoal - next_ag))
                reward -= action_penalty_coeff * np.square(action).sum()
                return reward * scale

        elif not absolute_goal and binary_reward:
            def controller_reward(ag, subgoal, next_ag, scale, action):
                reward_ctrl = action_penalty_coeff * -np.square(action).sum()
                fail = True
                if np.sqrt(np.sum(np.square(ag + subgoal - next_ag))) <= distance_threshold:
                    fail = False
                reward = reward_ctrl - float(fail)
                return reward * scale
    else:
        raise NotImplementedError
    return controller_reward


def is_goal_unreachable(a_net, state, goal, goal_dim, margin, device, absolute_goal=False):
    state = torch.from_numpy(state[:goal_dim]).float().to(device)
    goal = torch.from_numpy(goal).float().to(device)
    if not absolute_goal:
        goal = state + goal
    inputs = torch.stack((state, goal), dim=0)
    outputs = a_net(inputs)
    s_embedding = outputs[0]
    g_embedding = outputs[1]
    dist = F.pairwise_distance(s_embedding.unsqueeze(0), g_embedding.unsqueeze(0)).squeeze()
    return dist > margin


@total_ordering
class StorageElement:
    def __init__(self, state, achieved_goal, score):
        self.state = state
        self.achieved_goal = achieved_goal
        self.score = score

    def __eq__(self, other):
        return np.isclose(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return hash(tuple(self.state))


def unravel_elems(elems):
    return tuple(map(list, zip(*[(elem.state, elem.score) for elem in elems])))


def unravel_elems_ag(elems):
    return tuple(map(list, zip(*[(elem.achieved_goal, elem.score) for elem in elems])))


class PriorityQueue:
    def __init__(self, top_k, close_thr=0.1, discard_by_anet=False):
        self.elems = []
        self.elems_state_tensor = None
        self.elems_achieved_goal_tensor = None

        self.top_k = top_k
        self.close_thr = close_thr
        self.discard_by_anet = discard_by_anet

    def __len__(self):
        return len(self.elems)

    def add_list(self, state_list, achieved_goal_list, score_list, a_net=None):
        if self.discard_by_anet:
            self.discard_out_of_date_by_anet(achieved_goal_list, a_net)
        else:
            self.discard_out_of_date(achieved_goal_list)
        # total_timesteps = len(state_list)
        # Fill inf in the future observation | achieved goal
        new_elems = [StorageElement(state=state, achieved_goal=achieved_goal, score=score)
                     for timestep, (state, achieved_goal, score)
                     in enumerate(zip(state_list, achieved_goal_list, score_list))]
        self.elems.extend(new_elems)
        self.elems = list(set(self.elems))
        self.update_tensors()

    def update_tensors(self):
        self.elems_state_tensor = torch.FloatTensor(np.array([elems.state for elems in self.elems])).to(device)
        self.elems_achieved_goal_tensor = torch.FloatTensor(np.array([elems.achieved_goal for elems in self.elems])).to(device)

    # update novelty of similar states existing in storage to the newly encountered one.
    def discard_out_of_date(self, achieved_goal_list):
        if len(self.elems) == 0:
            return

        achieved_goals = torch.FloatTensor(np.array(achieved_goal_list)).to(device)
        dist = torch.cdist(self.elems_achieved_goal_tensor, achieved_goals)
        close = dist < self.close_thr
        keep = close.sum(dim=1) == 0
        self.elems = list(compress(self.elems, keep))
        self.update_tensors()

    def discard_out_of_date_by_anet(self, achieved_goal_list, a_net):
        assert a_net is not None
        if len(self.elems) == 0:
            return

        with torch.no_grad():
            achieved_goals = torch.FloatTensor(np.array(achieved_goal_list)).to(device)
            dist1 = torch.cdist(a_net(achieved_goals), a_net(self.elems_achieved_goal_tensor)).T
            dist2 = torch.cdist(a_net(self.elems_achieved_goal_tensor), a_net(achieved_goals))
            dist = (dist1 + dist2)/2

            close = dist < self.close_thr
            keep = close.sum(dim=1) == 0
            self.elems = list(compress(self.elems, keep))
            self.update_tensors()

    def get_elems(self):
        return unravel_elems(self.elems[:self.top_k])

    def get_states(self):
        return self.elems_state_tensor[:self.top_k]

    def get_landmarks(self):
        return self.elems_achieved_goal_tensor[:self.top_k]

    def squeeze_by_kth(self, k):
        k = min(k, len(self.elems))
        self.elems = sorted(self.elems, reverse=True)[:k]
        self.update_tensors()
        return self.elems[-1].score

    def squeeze_by_thr(self, thr):
        self.elems = sorted(self.elems, reverse=True)
        k = next((i for i, elem in enumerate(self.elems) if elem.score < thr), len(self.elems))

        self.elems = self.elems[:k]
        self.update_tensors()
        return unravel_elems(self.elems)

    def sample_batch(self, batch_size):
        sampled_elems = random.choices(population=self.elems, k=batch_size)
        return unravel_elems(sampled_elems)

    def save_log(self, timesteps, log_file):
        elems = self.get_elems()
        output_df = pd.DataFrame((timesteps, score, state) for state, score in zip(elems[0], elems[1]))
        output_df.to_csv(log_file, mode='a', header=False)

    def sample_by_novelty_weight(self):
        raise NotImplementedError

    def save(self, file):
        np.savez_compressed(file, elems=self.elems, top_k=self.top_k, close_thr=self.close_thr)

    def load(self, file):
        with np.load(file, allow_pickle=True) as data:
            self.elems = data['elems']
            self.top_k = data['top_k']
            self.close_thr = data['close_thr']


class TrajRwdQueue:

    def __init__(self, traj_max_num=2000):
        self.elems = []
        # rewards
        self.G = []
        # episode lengths
        self.T = []
        # s(t0)
        self.s0 = []
        # goals
        self.goals = []
        self.elems_state_tensor = None
        self.elems_achieved_goal_tensor = None

        self.max_size = traj_max_num

    def __len__(self):
        return len(self.elems)

    def add(self, state_traj, achieved_goal_traj, reward, goal=None):
        new_elems = StorageElement(state=state_traj, achieved_goal=achieved_goal_traj, score=reward)
        if len(self.elems) >= self.max_size:
            s0g = np.hstack([state_traj[0], goal])
            s0gs = np.hstack([self.s0, self.goals])
            s0gs_mse_sorti = np.argsort(np.sum((s0gs - s0g) ** 2, axis=-1))
            s0gs_mse_sorti = s0gs_mse_sorti[:int(self.max_size * 0.1)]
            replaced_i = np.argsort(np.array(self.G)[s0gs_mse_sorti])[0]
            replaced_i = s0gs_mse_sorti[replaced_i]
            if self.G[replaced_i] <= reward:
                self.elems[replaced_i] = new_elems
                self.G[replaced_i] = reward
                self.T[replaced_i] = len(state_traj)
                self.s0[replaced_i] = state_traj[0]
                self.goals[replaced_i] = goal
            else:
                # drop out this record.
                pass
        else:
            self.elems.append(new_elems)
            self.G.append(reward)
            self.T.append(len(state_traj))
            self.s0.append(state_traj[0])
            self.goals.append(goal)

    def get_states(self, sample_num=1e4):
        _sorted_elems = sorted(self.elems, reverse=True)
        _states, _ = unravel_elems(_sorted_elems)
        _states = [item for trajs in _states for item in trajs]
        _ags, _ = unravel_elems_ag(_sorted_elems)
        _ags = [item for trajs in _ags for item in trajs]
        self.elems_state_tensor = torch.FloatTensor(np.array(_states)[:int(sample_num)]).to(device)
        self.elems_achieved_goal_tensor = torch.FloatTensor(np.array(_ags)[:int(sample_num)]).to(device)

        return self.elems_state_tensor

    def get_landmarks(self, sample_num=1e4):
        return self.elems_achieved_goal_tensor

    def save(self, file):
        np.savez_compressed(file, max_size=self.max_size, elems=self.elems,
                            G=self.G, T=self.T, s0=self.s0, goals=self.goals)

    def load(self, file):
        with np.load(file, allow_pickle=True) as data:
            self.max_size = data['max_size']
            self.elems = data['elems']
            self.G = data['G']
            self.T = data['T']
            self.s0 = data['s0']
            self.goals = data['goals']


class TrajRwdQueueRW(TrajRwdQueue):

    def __init__(self, traj_max_num=2000, alpha=0.1, mix_mode='HR'):
        self.alpha = alpha, # Temperature of the softmax. The higher, the closer to uniform distribution.
        self.mix_mode = mix_mode # TopK, HR, RW, {TopK} U {HR}, {Uniform} U {HR}, {TopK} U {Uniform} U {HR}
        super().__init__(traj_max_num)

    def get_states(self, sample_num=1e4):
        if 'TopK' in self.mix_mode:
            super().get_states(sample_num)
        if 'HR' in self.mix_mode or 'RW' in self.mix_mode:
            _sample_probs = self._compute_sample_probs()
            _states, _ = unravel_elems(self.elems)
            _states = [item for trajs in _states for item in trajs]
            _ags, _ = unravel_elems_ag(self.elems)
            _ags = [item for trajs in _ags for item in trajs]
            _cache_indices = np.random.choice(
                range(len(_states)),
                sample_num,
                p=_sample_probs)
            if 'TopK' in self.mix_mode:
                self.elems_state_tensor = torch.cat((torch.FloatTensor(np.array(_states)[_cache_indices]).to(device), self.elems_state_tensor), dim=0)
                self.elems_achieved_goal_tensor = torch.cat((torch.FloatTensor(np.array(_ags)[_cache_indices]).to(device), self.elems_achieved_goal_tensor), dim=0)
            else:
                self.elems_state_tensor = torch.FloatTensor(np.array(_states)[_cache_indices]).to(device)
                self.elems_achieved_goal_tensor = torch.FloatTensor(np.array(_ags)[_cache_indices]).to(device)
        return self.elems_state_tensor

    def _compute_sample_probs(self):
        G = self.G
        T = self.T
        G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
        w_it = (G_it - G_it.min()) / (G_it.max() - G_it.min())
        w_it = softmax(G_it / self.alpha)
        return w_it


class TrajRwdQueueHR(TrajRwdQueueRW):

    def __init__(self, traj_max_num=2000, alpha=0.1, mix_mode='HR'):
        super().__init__(traj_max_num, alpha, mix_mode)

    def _compute_sample_probs(self, n_jobs=16):
        G = self.G
        T = self.T
        G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
        s0g = np.hstack([np.array(self.s0)[..., :3], self.goals])
        # TODO: Consider utilizing a nonlinear estimator!!!
        V = LinearRegression(n_jobs=n_jobs).fit(s0g, G).predict(s0g)
        V_it = np.asarray(reduce(lambda x, y: x + y, [[V_i] * T_i for V_i, T_i in zip(V, T)]))
        A_it = G_it - V_it
        A_it = (A_it - A_it.min()) / np.clip(A_it.max() - A_it.min(), a_min=1e-6, a_max=None)
        w_it = softmax(A_it / self.alpha)
        w_it /= w_it.sum() # Avoid numerical errors
        return w_it


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class AutoLambda(object):
    def __init__(self, init_value, update_rate):
        self.init_value = init_value
        assert 0 <= update_rate <= 1, 'The update rate should be in the range of 0 to 1!'
        self.update_rate = update_rate
        self._value = init_value

    def update(self, x):
        self._value = self.update_rate * x + (1 - self.update_rate) * self._value
        return self._value

    @property
    def value(self):
        return self._value

    @property
    def enable(self):
        return self.init_value != 0 or self.update_rate != 0

    @property
    def is_dynamic(self):
        return self.update_rate != 0

class LossesList:

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = dict()
        self._len = 0

    def __len__(self):
        return self._len

    def push(self, key, value):
        if not (key in self.losses and isinstance(self.losses[key], list)):
            self.losses[key] = list()
        if torch.is_tensor(value):
            value = value.cpu().data.numpy().mean()
        self.losses[key].append(value)
        self._len += 1
        return self.losses[key]

    def _mean(self, key):
        if key is None or not (key in self.losses and isinstance(self.losses[key], list)):
            return 0
        else:
            return np.mean(self.losses[key])

    def mean(self, key=None):
        if key is None:
            return {_key: self._mean(key) for _key in self.losses}
        else:
            return self._mean(key)