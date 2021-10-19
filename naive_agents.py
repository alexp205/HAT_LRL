import random
import numpy as np
import pickle
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import mse_loss

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.task_ids = []
        self.curr_task_id = -1
        self.task_shapes = {}
        self.fc_ins = nn.ModuleList()
        self.fc_outs = nn.ModuleList()
        #self.hl_size = 2048 # TODO 2048?
        self.hl_size = 128
        #self.hl_size = 512

        self.relu = nn.ReLU()

    def task_configure(self, task_id, s_shape, a_shape):
        if task_id not in self.task_ids:
            if 3 == len(s_shape):
                #fc_in = nn.ModuleList()
                #fc_in.append(nn.Conv2d(1, 16, 4, 2, 1))
                #fc_in.append(nn.ReLU())
                #fc_in.append(nn.MaxPool2d((2, 2)))
                #fc_in.append(nn.Conv2d(16, 16, 4, 2, 1))
                #fc_in.append(nn.ReLU())
                #fc_in.append(nn.MaxPool2d((2, 2)))
                #fc_in.append(nn.Flatten())
                #fc_in.append(nn.Linear(400,self.hl_size))
                #self.fc_ins.append(fc_in)
                # TODO maybe try batch norm as well
                fc_in = nn.ModuleList()
                fc_in.append(nn.Conv2d(1, 32, 8, 4))
                #fc_in.append(nn.BatchNorm2d(32))
                fc_in.append(nn.ReLU())
                fc_in.append(nn.Conv2d(32, 64, 4, 2))
                #fc_in.append(nn.BatchNorm2d(64))
                fc_in.append(nn.ReLU())
                fc_in.append(nn.Conv2d(64, 64, 3, 1))
                #fc_in.append(nn.BatchNorm2d(64))
                fc_in.append(nn.ReLU())
                fc_in.append(nn.Flatten())
                fc_in.append(nn.Linear(2304,self.hl_size))
                self.fc_ins.append(fc_in)
            else:
                self.fc_ins.append(nn.Linear(s_shape[0], self.hl_size))
            self.fc_outs.append(nn.Linear(self.hl_size, a_shape[0]))

            self.task_ids.append(task_id)
            self.task_shapes[task_id] = len(s_shape)
        self.curr_task_id = task_id

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #print(observations.shape)

        x = observations
        #x = observations.cpu()

        #print(x.device)

        if 3 == self.task_shapes[self.curr_task_id]:
            for fc_in in self.fc_ins[self.curr_task_id]:
                x = fc_in(x)
                #print(x.shape)
        else:
            x = self.fc_ins[self.curr_task_id](x)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.fc2(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.fc3(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.fc_outs[self.curr_task_id](x)
        #print(x.shape)
        #input("wait")

        return x

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'task_ids': self.task_ids,
            'task_shapes': self.task_shapes,
            'fc_ins': self.fc_ins,
            'fc_outs': self.fc_outs
        }, path)

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        self.task_ids = checkpoint['task_ids']
        self.task_shapes = checkpoint['task_shapes']
        self.fc_ins = checkpoint['fc_ins']
        self.fc_outs = checkpoint['fc_outs']

class PlayerAgent:
    def __init__(self, gamma, alpha, seed_val, mem_len, epsilon, epsilon_decay, epsilon_min):
        self.gamma = gamma
        self.alpha = alpha
        self.seed_val = seed_val
        self.counter = 0
        self.task_counters = {}
        self.s_shape = (-2)
        self.a_shape = (-2)
        self.task_id = -1
        self.is_first_task = True

        self.model = ValueNet().cuda()
        self.target_model = ValueNet().cuda()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.memory = {} # set of mems, one per task

        self.mem_len = mem_len
        self.epsilon = {}
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def set_up_task(self, task_id, s_shape, a_shape):
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.model.task_configure(task_id, s_shape, a_shape)
        if self.target_model.state_dict().keys() != self.model.state_dict().keys():
            temp_model = copy.deepcopy(self.target_model)
            self.target_model = copy.deepcopy(self.model)
            for name, param in self.target_model.named_parameters():
                for t_name, t_param in temp_model.named_parameters():
                    if name == t_name:
                        param.data.copy_(t_param.data)
        self.target_model.task_configure(task_id, s_shape, a_shape)

        self.task_id = task_id
        if task_id not in self.memory:
            self.memory[task_id] = []
            self.task_counters[task_id] = 0
            self.epsilon[task_id] = self.epsilon_init
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        self.target_model.cuda()
        self.model.cuda()

    def store_experience(self, s, a, r, s_prime, done, ep, run):
        s_in = s.tolist()
        s_prime_in = s_prime.tolist()

        mem = self.memory[self.task_id]

        if len(mem) < self.mem_len[self.task_id]:
            mem.append((s_in, a, r, s_prime_in, done, ep, run))
        else:
            #replace_idx = random.randrange(len(mem)) # rand
            replace_idx = self.counter % self.mem_len[self.task_id] # wrap
            mem[replace_idx] = (s_in, a, r, s_prime_in, done, ep, run)

        self.counter += 1
        self.task_counters[self.task_id] += 1

    def rescale(self, val, in_min, in_max, out_min, out_max):
        return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

    def sigmoid(self, val):
        return 1. / (1. + np.exp(-val))

    def update_models(self, batch_size, update_num, num_updates):
        record_range = min(self.task_counters[self.task_id], self.mem_len[self.task_id])
        batch_indices = np.random.choice(record_range, batch_size)
        batch = np.asarray(self.memory[self.task_id], dtype=object)[batch_indices] # TODO do we replay data from ALL tasks?

        # ref: self.memory.append((s_in, a, r, s_prime_in, done, ep, run))
        state = torch.from_numpy(np.asarray(batch[:,0].tolist())).float()
        action = torch.from_numpy(np.asarray(batch[:,1].tolist())).float()
        reward = torch.from_numpy(np.asarray(batch[:,2].tolist())).float()
        #print("CHECK state shape")
        #print(np.asarray(batch[:,3].tolist()).shape)
        state_new = torch.from_numpy(np.asarray(batch[:,3].tolist())).float()
        terminal = torch.from_numpy(np.asarray(batch[:,4].tolist())).float()
        state = Variable(state).cuda()
        action = Variable(action).cuda()
        #print("CHECK action")
        #print(action)
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()
        self.model.eval()
        self.target_model.eval()

        action_new = self.model.forward(state_new)
        #print("action_new")
        #print(action_new)
        action_new = action_new.max(dim=1)[1].cpu().data.view(-1, 1)
        #print("action_new")
        #print(action_new)
        action_new_onehot = torch.zeros(batch_size, self.a_shape[0])
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0)).cuda()

        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        action_target = self.target_model.forward(state_new)
        #print("action_target")
        #print(action_target)
        action_target = action_target*action_new_onehot
        #print("action_target")
        #print(action_target)
        y = reward + torch.mul((action_target.sum(dim=1)*(1-terminal)), self.gamma)
        #print("y")
        #print(y)

        ref_action = torch.unsqueeze(action.long().cpu(), 1)
        ref_action_onehot = torch.zeros(batch_size, self.a_shape[0])
        #print("ref_action_onehot")
        #print(ref_action_onehot.shape)
        #print("ref_action")
        #print(ref_action.shape)
        ref_action_onehot = Variable(ref_action_onehot.scatter_(1, ref_action, 1.0)).cuda()

        # regression Q(s, a) -> y
        self.model.train()
        action_old = self.model.forward(state)
        #print("action_old")
        #print(action_old)
        action_old = action_old*ref_action_onehot
        #print("action_old")
        #print(action_old)
        Q = action_old.sum(dim=1)
        #print("Q")
        #print(Q)
        #input("wait")
        loss = mse_loss(input=Q, target=y.detach())

        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        temp_epsilon = self.epsilon[self.task_id]
        temp_epsilon *= self.epsilon_decay
        self.epsilon[self.task_id] = max(self.epsilon_min, temp_epsilon)

    def update_target_models(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)

    def act(self, s_sample):
        a = None

        # with epsilon prob to choose random action else choose argmax Q estimate action
        if np.random.rand() <= self.epsilon[self.task_id]:
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
