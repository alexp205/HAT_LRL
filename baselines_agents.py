import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import mse_loss
import math

# non-img
class ValueNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(ValueNet, self).__init__()
        # LunarLander
        # ref: https://github.com/frizner/LunarLander-v2/blob/master/lunarlander.py
        #self.fc1 = nn.Linear(s_shape[0], 128)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, a_shape[0])
        # gridworld
        # ref: https://github.com/mingen-pan/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/blob/master/main.py
        #self.fc1 = nn.Linear(s_shape[0], 128)
        #self.fc2 = nn.Linear(128, 128)
        #self.fc3 = nn.Linear(128, a_shape[0])
        # CarPole
        # ref: https://github.com/gsurma/cartpole/blob/master/cartpole.py
        self.fc1 = nn.Linear(s_shape[0], 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, a_shape[0])
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.fc1(observations)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
# img

class PlayerAgent:
    def __init__(self, gamma, alpha, seed_val, mem_len, epsilon, epsilon_decay, epsilon_min, s_shape, a_shape):
        self.gamma = gamma
        self.alpha = alpha
        self.seed_val = seed_val
        self.counter = 0
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.model = ValueNet(self.s_shape, self.a_shape).cuda()
        self.target_model = ValueNet(self.s_shape, self.a_shape).cuda()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.memory = []
        self.target_model.cuda()
        self.model.cuda()

        self.mem_len = mem_len
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def store_experience(self, s, a, r, s_prime, done, ep, run):
        s_in = s.tolist()
        s_prime_in = s_prime.tolist()

        mem = self.memory

        if len(mem) < self.mem_len:
            mem.append((s_in, a, r, s_prime_in, done, ep, run))
        else:
            #replace_idx = random.randrange(len(mem)) # rand
            replace_idx = self.counter % self.mem_len # wrap
            mem[replace_idx] = (s_in, a, r, s_prime_in, done, ep, run)

        self.counter += 1

    def rescale(self, val, in_min, in_max, out_min, out_max):
        return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

    def sigmoid(sefl, val, scale):
        return 1. / (1. + np.exp(-scale * val))

    def update_models(self, batch_size, do_val_update, do_perm_update):
        record_range = min(self.counter, self.mem_len)
        batch_indices = np.random.choice(record_range, batch_size)
        batch = np.asarray(self.memory, dtype=object)[batch_indices]

        state = torch.from_numpy(np.asarray(batch[:,0].tolist())).float()
        action = torch.from_numpy(np.asarray(batch[:,1].tolist())).float()
        reward = torch.from_numpy(np.asarray(batch[:,2].tolist())).float()
        state_new = torch.from_numpy(np.asarray(batch[:,3].tolist())).float()
        terminal = torch.from_numpy(np.asarray(batch[:,4].tolist())).float()
        state = Variable(state).cuda()
        action = Variable(action).cuda()
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()
        self.model.eval()
        self.target_model.eval()

        action_new = self.model.forward(state_new)
        action_new = action_new.max(dim=1)[1].cpu().data.view(-1, 1)
        action_new_onehot = torch.zeros(batch_size, self.a_shape[0])
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0)).cuda()

        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        action_target = self.target_model.forward(state_new)
        action_target = action_target*action_new_onehot
        y = reward + torch.mul((action_target.sum(dim=1)*(1-terminal)), self.gamma)

        ref_action = torch.unsqueeze(action.long().cpu(), 1)
        ref_action_onehot = torch.zeros(batch_size, self.a_shape[0])
        ref_action_onehot = Variable(ref_action_onehot.scatter_(1, ref_action, 1.0)).cuda()

        # regression Q(s, a) -> y
        self.model.train()
        action_old = self.model.forward(state)
        action_old = action_old*ref_action_onehot
        Q = action_old.sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())

        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        temp_epsilon = self.epsilon
        temp_epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, temp_epsilon)

    def update_target_models(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)

    def act(self, s_sample):
        a = None

        # with epsilon prob to choose random action else choose argmax Q estimate action
        if np.random.rand() <= self.epsilon:
            a = random.randint(0, self.a_shape[0]-1)
        else:
            state = torch.from_numpy(np.expand_dims(s_sample, axis=0)).float()
            state = Variable(state).cuda()

            self.model.eval()
            estimate = self.model.forward(state)
            estimate = estimate.max(dim=1)
            a = estimate[1].data[0].cpu().numpy()

        return a

    def load(self):
        print("loading model!")
        if self.model:
            self.model.load("./save/model" + str(self.seed_val) + ".h5", self.optimizer)
        if self.target_model:
            self.target_model.load("./save/target_model" + str(self.seed_val) + ".h5", self.optimizer)
        #with open("./save/memory.pickle", "rb") as h:
        #    self.memory = pickle.load(h)

    def save(self):
        print("saving model!")
        if self.model:
            self.model.save("./save/model" + str(self.seed_val) + ".h5", self.counter, self.optimizer)
        if self.target_model:
            self.target_model.save("./save/target_model" + str(self.seed_val) + ".h5", self.counter, self.optimizer)
        #with open("./save/memory.pickle", "wb") as h:
        #    pickle.dump(self.memory, h)
