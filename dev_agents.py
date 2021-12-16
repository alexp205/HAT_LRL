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
from collections import deque

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.task_ids = []
        self.curr_task_id = -1
        self.task_shapes = {}
        self.fc_ins = nn.ModuleList()
        self.fc_outs = nn.ModuleList()
        #self.hl_size = 512
        self.hl_size = 1024

        self.fc1 = nn.Linear(self.hl_size, self.hl_size)
        self.efc1 = nn.ModuleList()
        #self.s_efc1 = nn.ModuleList()
        #self.fc2 = nn.Linear(self.hl_size, self.hl_size)
        #self.efc2 = nn.ModuleList()
        ##self.s_efc2 = nn.ModuleList()
        #self.fc3 = nn.Linear(self.hl_size, self.hl_size)
        #self.efc3 = nn.ModuleList()
        ##self.s_efc3 = nn.ModuleList()

        self.relu = nn.ReLU()
        self.gate = nn.Sigmoid()

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
                #temp_in = nn.Linear(s_shape[0], self.hl_size)
                #for p in temp_in.parameters():
                #    p.requires_grad = False
                #self.fc_ins.append(temp_in)
            self.fc_outs.append(nn.Linear(self.hl_size, a_shape[0]))
            #temp_out = nn.Linear(self.hl_size, a_shape[0])
            #for p in temp_out.parameters():
            #    p.requires_grad = False
            #self.fc_outs.append(temp_out)

            self.efc1.append(nn.Embedding(1, self.hl_size))
            #self.efc2.append(nn.Embedding(1, self.hl_size))
            #self.efc3.append(nn.Embedding(1, self.hl_size))
            #self.s_efc1.append(nn.Embedding(1, self.hl_size))
            #self.s_efc2.append(nn.Embedding(1, self.hl_size))
            #self.s_efc3.append(nn.Embedding(1, self.hl_size))

            self.task_ids.append(task_id)
            self.task_shapes[task_id] = len(s_shape)
        self.curr_task_id = task_id

    # TODO need dropout?
    #def forward(self,t,x,s=1):
    #    # Gates
    #    masks=self.mask(t,s=s)
    #    if self.nlayers==1:
    #        gfc1=masks
    #    elif self.nlayers==2:
    #        gfc1,gfc2=masks
    #    elif self.nlayers==3:
    #        gfc1,gfc2,gfc3=masks
    #    # Gated
    #    h=self.drop1(x.view(x.size(0),-1))
    #    h=self.drop2(self.relu(self.fc1(h)))
    #    h=h*gfc1.expand_as(h)
    #    if self.nlayers>1:
    #        h=self.drop2(self.relu(self.fc2(h)))
    #        h=h*gfc2.expand_as(h)
    #        if self.nlayers>2:
    #            h=self.drop2(self.relu(self.fc3(h)))
    #            h=h*gfc3.expand_as(h)
    #    y=[]
    #    for t,i in self.taskcla:
    #        y.append(self.last[t](h))
    #    return y,masks
    def forward(self, observations: torch.Tensor, s=1) -> torch.Tensor:
        # gates
        masks = self.mask(self.curr_task_id, s)
        #gfc1, gfc2, gfc3 = masks
        #gfc1, gfc2 = masks
        gfc1, = masks

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
        x = self.relu(x)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        #print(x.shape)
        x = x * gfc1.expand_as(x)
        #print(x.shape)
        #x = self.fc2(x)
        #x = self.relu(x)
        ##print(x.shape)
        #x = x * gfc2.expand_as(x)
        ##print(x.shape)
        #x = self.fc3(x)
        #x = self.relu(x)
        ##print(x.shape)
        #x = x * gfc3.expand_as(x)
        ##print(x.shape)
        x = self.fc_outs[self.curr_task_id](x)
        #print(x.shape)
        #input("wait")

        return x, masks

    def mask(self, task_id, s=1):
        temp_id = Variable(torch.LongTensor([0]), volatile=False).cuda()
        gfc1 = self.gate(s*self.efc1[self.curr_task_id](temp_id))
        #gfc2 = self.gate(s*self.efc2[self.curr_task_id](temp_id))
        #gfc3 = self.gate(s*self.efc3[self.curr_task_id](temp_id))
        #gfc1 = self.gate(self.s_efc1[self.curr_task_id](temp_id)) * self.gate(s*self.efc1[self.curr_task_id](temp_id))
        #gfc2 = self.gate(self.s_efc2[self.curr_task_id](temp_id)) * self.gate(s*self.efc2[self.curr_task_id](temp_id))
        #gfc3 = self.gate(self.s_efc3[self.curr_task_id](temp_id)) * self.gate(s*self.efc3[self.curr_task_id](temp_id))
        #return [gfc1, gfc2, gfc3]
        #return [gfc1, gfc2]
        return [gfc1]

    def get_view_for(self, n, masks):
        #gfc1, gfc2, gfc3 = masks
        #gfc1, gfc2 = masks
        gfc1, = masks
        if n == 'fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n == 'fc1.bias':
            return gfc1.data.view(-1)
        elif n == 'fc2.weight':
            post = gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre = gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n == 'fc2.bias':
            return gfc2.data.view(-1)
        elif n == 'fc3.weight':
            post = gfc3.data.view(-1,1).expand_as(self.fc3.weight)
            pre = gfc2.data.view(1,-1).expand_as(self.fc3.weight)
            return torch.min(post,pre)
        elif n == 'fc3.bias':
            return gfc3.data.view(-1)
        return None

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'task_ids': self.task_ids,
            'task_shapes': self.task_shapes,
            'fc_ins': self.fc_ins,
            'fc_outs': self.fc_outs,
            'efc1': self.efc1
            #'efc2': self.efc2
            #'efc3': self.efc3,
            #'s_efc1': self.s_efc1,
            ##'s_efc3': self.s_efc3,
            #'s_efc2': self.s_efc2
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
        self.efc1 = checkpoint['efc1']
        #self.efc2 = checkpoint['efc2']
        #self.efc3 = checkpoint['efc3']
        #self.s_efc1 = checkpoint['s_efc1']
        ##self.s_efc3 = checkpoint['s_efc3']
        #self.s_efc2 = checkpoint['s_efc2']

class PlayerAgent:
    def __init__(self, gamma, alpha, seed_val, mem_len, epsilon, epsilon_decay, epsilon_min, smax, lamb, thresh_emb, thresh_cosh, clipgrad, surv_len, reg_mod, reg_alpha, reg_temperature_decay):
        self.gamma = gamma
        self.alpha = alpha
        self.seed_val = seed_val
        self.counter = 0
        self.task_counters = {}
        self.s_shape = (-2)
        self.a_shape = (-2)
        #self.max_s_shape = -1
        #self.max_a_shape = -1
        self.task_id = -1
        self.is_first_task = True

        self.s_factor = -1.
        self.mask_pre = None
        self.mask_back = None

        self.model = ValueNet().cuda()
        self.target_model = ValueNet().cuda()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.memory = {} # set of mems, one per task
        self.surveyor = {}

        self.mem_len = mem_len
        self.epsilon = {}
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.smax = smax
        self.lamb = lamb
        self.thresh_emb = thresh_emb
        self.thresh_cosh = thresh_cosh
        self.clipgrad = clipgrad
        self.surv_len = surv_len
        self.reg_mod = reg_mod

        self.reg_scaler = {}
        self.reg_temperature = {}
        self.reg_scaler_init = 0.0025
        self.reg_temperature_init = 1.
        self.reg_alpha = reg_alpha
        self.reg_temperature_decay = reg_temperature_decay

        self.task_perfs = {}

    def set_up_task(self, task_id, s_shape, a_shape):
        #print("in set_up_task")
        #print("target_model params")
        #print(self.target_model.parameters())
        #print(self.target_model.state_dict())
        #print("model params")
        #print(self.model.parameters())
        #print(self.model.state_dict())
        #print("checking equality")
        #print(self.target_model.state_dict().keys() == self.model.state_dict().keys())
        #input("wait")
        self.s_shape = s_shape
        self.a_shape = a_shape
        #self.max_s_shape = max(self.max_s_shape, np.prod(s_shape))
        #self.max_a_shape = max(self.max_a_shape, sum(a_shape))

        self.model.task_configure(task_id, s_shape, a_shape)
        if self.target_model.state_dict().keys() != self.model.state_dict().keys():
            print("--- DOING PARAM COPY ---")
            temp_model = copy.deepcopy(self.target_model)
            self.target_model = copy.deepcopy(self.model)
            for name, param in self.target_model.named_parameters():
                for t_name, t_param in temp_model.named_parameters():
                    if name == t_name:
                        param.data.copy_(t_param.data)
        self.target_model.task_configure(task_id, s_shape, a_shape) # TODO needed?

        self.task_id = task_id
        if task_id not in self.memory:
            self.memory[task_id] = []
            self.task_counters[task_id] = 0
            self.epsilon[task_id] = self.epsilon_init
            self.surveyor[task_id] = deque(maxlen=self.surv_len[task_id])
            self.task_perfs[task_id] = []
            self.reg_scaler[task_id] = self.reg_scaler_init
            self.reg_temperature[task_id] = self.reg_temperature_init
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        self.target_model.cuda()
        self.model.cuda()

    def store_experience(self, s, a, r, s_prime, done, ep, run, ep_r):
        s_in = s.tolist()
        s_prime_in = s_prime.tolist()

        mem = self.memory[self.task_id]

        if len(mem) < self.mem_len[self.task_id]:
            mem.append((s_in, a, r, s_prime_in, done, ep, run, ep_r))
        else:
            #replace_idx = random.randrange(len(mem)) # rand
            replace_idx = self.counter % self.mem_len[self.task_id] # wrap
            mem[replace_idx] = (s_in, a, r, s_prime_in, done, ep, run, ep_r)

        surv = self.surveyor[self.task_id]
        if not any((s == elem).all() for elem in surv):
            surv.appendleft(s.tolist())

        self.counter += 1
        self.task_counters[self.task_id] += 1

    def rescale(self, val, in_min, in_max, out_min, out_max):
        return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

    def sigmoid(self, val):
        return 1. / (1. + np.exp(-val))
    def pseudosigmoid(self, val):
        #print("val:")
        #print(val)
        #return 1. - np.exp(-val)
        return 1. - np.exp(-10*val)

    def get_traj(self, traj_id, prev_idx):
        traj = []

        temp_mem = np.roll(np.asarray(self.memory[self.task_id], dtype=object), -(prev_idx+1), axis=0)

        for trans in reversed(temp_mem):
            idx = trans[5]
            if traj_id == idx:
                traj.append(trans)
                if 1 == trans[6]:
                    break
        traj = np.asarray(traj, dtype=object)

        traj = traj[traj[:,6].argsort()]

        return traj

    def get_traj_perf(self, traj_id, prev_idx):
        traj_perf = 0.

        temp_mem = np.roll(np.asarray(self.memory[self.task_id], dtype=object), -(prev_idx+1), axis=0)

        for trans in reversed(temp_mem):
            idx = trans[5]
            if traj_id == idx:
                traj_perf = trans[7]
                break

        return traj_perf

    def store_perf(self, ep_perf):
        self.task_perfs[self.task_id].append(ep_perf)

    def get_masks(self):
        mask = self.model.mask(self.task_id, self.smax)
        for i in range(len(mask)):
            mask[i] = Variable(mask[i].data.clone(), requires_grad=False)
        if self.is_first_task:
            self.mask_pre = mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i] = torch.max(self.mask_pre[i], mask[i])

        self.mask_back = {}
        for n,_ in self.model.named_parameters():
            vals = self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals

        #print("mask_pre")
        #print(self.mask_pre)
        #print("mask_back")
        #print(self.mask_back)
        #input("wait")

    def get_s_comps(self, update_num, num_updates):
        comps = []

        # time comp
        #time_comp = update_num/num_updates
        #comps.append(time_comp)

        # diversity comp
        div_comp = -1.

        exps = list(self.surveyor[self.task_id])
        #print("exps:")
        #print(exps)
        div = np.std(exps, axis=0)
        #print("div:")
        #print(div)
        div = 1. - self.pseudosigmoid(np.mean(div)) # TODO hyperparam (in fxn)...
        #print("div:")
        #print(div)
        #input("wait")

        div_comp = div
        comps.append(div_comp)

        # perf comp
        #perf_comp = -1.

        ##prev_idx = (self.task_counters[self.task_id]-1) % self.mem_len[self.task_id]
        ##traj_id = self.memory[self.task_id][prev_idx][5]
        ###print("traj_id:")
        ###print(traj_id)
        ##perfs = []
        ##for i in range(max(0,traj_id-5), traj_id+1):
        ##    #print("i:")
        ##    #print(i)
        ##    #traj = np.asarray(self.get_traj(i, prev_idx), dtype=object)
        ##    #perfs.append(sum(traj[:,2]))
        ##    perfs.append(self.get_traj_perf(i, prev_idx))
        ###input("wait")
        ###print("perfs:")
        ###print(perfs)
        ##perf_mean = np.mean(perfs)
        ###print("perf_mean:")
        ###print(perf_mean)
        ###input("wait")

        ##env_r_bounds = {0: [-1000, 300], 1: [-1, 1], 2: [0, 500]}
        ##perf_comp = self.rescale(perf_mean, env_r_bounds[self.task_id][0], env_r_bounds[self.task_id][1], 0., 1.) # TODO for now, we need to elim this reliance on domain knowledge somehow
        ###perf_comp = a
        ##comps.append(perf_comp)

        #temp_perfs = self.task_perfs[self.task_id]
        ##print("temp_perfs:")
        ##print(temp_perfs)

        #if len(temp_perfs) > 0:
        #    # sliding window
        #    avg_perfs = np.zeros_like(temp_perfs)
        #    for k, x in enumerate(temp_perfs):
        #        perfs_subarr = temp_perfs[max(0,k-50):k+1] # TODO hyperparam...
        #        mean_val = np.mean(perfs_subarr)
        #        avg_perfs[k] = mean_val
        #    #print("avg_perfs:")
        #    #print(avg_perfs)
        #    max_perf = max(avg_perfs)
        #    perf_improve = max_perf - avg_perfs[0]
        #    #print("max_perf:")
        #    #print(max_perf)
        #    #print("perf_improve:")
        #    #print(perf_improve)
        #    env_r_bounds = {0: [0, 1300], 1: [0, 2], 2: [0, 500]}
        #    perf_comp = self.rescale(perf_improve, env_r_bounds[self.task_id][0], env_r_bounds[self.task_id][1], 0., 1.) # TODO for now, we need to elim this reliance on domain knowledge somehow
        #    #print("perf_comp:")
        #    #print(perf_comp)
        #    #input("wait")
        #else:
        #    perf_comp = 0.
        #comps.append(perf_comp)

        return comps

    def update_reg_scaler(self):
        len_tp = len(self.task_perfs[self.task_id])
        mu_t_prev = sum(self.task_perfs[self.task_id][:len_tp-1])/(max(len_tp-1,1))
        mu_t = sum(self.task_perfs[self.task_id][:len_tp])/len_tp
        R_perf = mu_t - mu_t_prev

        if 0 == len_tp % 10:
            print("BEFORE reg_scaler:")
            print(self.reg_scaler[self.task_id])
            print("BEFORE reg_temperature:")
            print(self.reg_temperature[self.task_id])

        self.reg_scaler[self.task_id] = self.reg_scaler[self.task_id] - self.reg_alpha * self.reg_temperature[self.task_id] * R_perf

        self.reg_temperature[self.task_id] *= self.reg_temperature_decay

        if 0 == len_tp % 10:
            print("mu_t_prev:")
            print(mu_t_prev)
            print("mu_t:")
            print(mu_t)
            print("R_perf:")
            print(R_perf)
            print("AFTER reg_scaler:")
            print(self.reg_scaler[self.task_id])
            print("AFTER reg_temperature:")
            print(self.reg_temperature[self.task_id])
        #input("wait")

    def update_models(self, batch_size, update_num, num_updates):

        #print("in update_models")
        #print("target_model params")
        #print(self.target_model.parameters())
        #print(self.target_model.state_dict())
        #print("model params")
        #print(self.model.parameters())
        #print(self.model.state_dict())
        #print("checking equality")
        #print(self.target_model.state_dict().keys() == self.model.state_dict().keys())
        #input("wait")

        record_range = min(self.task_counters[self.task_id], self.mem_len[self.task_id])
        batch_indices = np.random.choice(record_range, batch_size)
        batch = np.asarray(self.memory[self.task_id], dtype=object)[batch_indices]

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

        self.s_factor = (self.smax-1/self.smax)*update_num/num_updates+1/self.smax # TODO same for RL? or do we need to modify? 
        #s_comps = self.get_s_comps(update_num, num_updates)
        ##print("s_comps:")
        ##print(s_comps)
        ##input("wait")
        #s_comp = self.rescale(sum(s_comps),0.,len(s_comps),0.,1.)
        ##print("s_comp:")
        ##print(s_comp)
        ##input("wait")
        #self.s_factor = (self.smax-1/self.smax)*s_comp+1/self.smax # TODO maybe change
        #if 0 == self.task_counters[self.task_id] % 1000:
        #    print("s: {}".format(self.s_factor))

        action_new, masks = self.model.forward(state_new, self.s_factor)
        #print("action_new")
        #print(action_new)
        #print("masks")
        #print(masks)
        action_new = action_new.max(dim=1)[1].cpu().data.view(-1, 1)
        #print("action_new")
        #print(action_new)
        action_new_onehot = torch.zeros(batch_size, self.a_shape[0])
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0)).cuda()

        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        action_target, _ = self.target_model.forward(state_new, self.s_factor)
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
        action_old, _ = self.model.forward(state, self.s_factor)
        #print("action_old")
        #print(action_old)
        action_old = action_old*ref_action_onehot
        #print("action_old")
        #print(action_old)
        Q = action_old.sum(dim=1)
        #print("Q")
        #print(Q)
        #input("wait")

        # TODO testing stuff: larger corrections for post-task differing Q-vals
        #if 0 == self.counter % 10:
        #    print("vals:")
        #    print(Q)
        #    print("rewards:")
        #    print(reward)
        #    print("target:")
        #    print(y)
        #    print("val-reward difference")
        #    print(torch.mean(torch.abs(Q-reward)))
        #    print("val-target difference")
        #    print(torch.mean(torch.abs(Q-y)))
        #    input("wait")

        #loss, _ = self.criterion(Q, y.detach(), masks)
        loss, _ = self.proto_criterion(Q, y.detach(), masks)

        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()

        # HAT - restrict layer grads
        if not self.is_first_task:
            for n,p in self.model.named_parameters():
                if n in self.mask_back:
                    p.grad.data *= self.mask_back[n]

        # HAT - compensate embedding grads
        for n,p in self.model.named_parameters():
            if n.startswith('e'):
                num = torch.cosh(torch.clamp(self.s_factor*p.data, -self.thresh_cosh, self.thresh_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.smax/self.s_factor*num/den

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
        self.optimizer.step()

        # HAT - constrain embeddings
        for n,p in self.model.named_parameters():
            if n.startswith('e'):
                p.data = torch.clamp(p.data, -self.thresh_emb, self.thresh_emb)

        temp_epsilon = self.epsilon[self.task_id]
        temp_epsilon *= self.epsilon_decay
        self.epsilon[self.task_id] = max(self.epsilon_min, temp_epsilon)

    def criterion(self, outputs, targets, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m,mp in zip(masks, self.mask_pre):
                aux = 1-mp
                reg += (m*aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count
        #print("outputs")
        #print(outputs)
        #print("targets")
        #print(targets)
        #print("reg")
        #print(reg)
        #print("count")
        #print(count)
        #input("wait")
        return mse_loss(input=outputs, target=targets.detach()) + self.lamb * reg, reg

    def proto_criterion(self, outputs, targets, masks):
        reg_sparse = 0
        count = 0
        if self.mask_pre is not None:
            for m,mp in zip(masks, self.mask_pre):
                aux = 1-mp
                reg_sparse += (m*aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg_sparse += m.sum()
                count += np.prod(m.size()).item()
        reg_sparse /= count
        #print("outputs")
        #print(outputs)
        #print("targets")
        #print(targets)
        #print("reg_sparse")
        #print(reg_sparse)
        #print("count")
        #print(count)
        #input("wait")

        reg_size = 0
        num_open_nodes = 0
        total_mask_size = 0
        for m in masks:
            #print("checking masks")
            #print(m)
            #print(m.sum())
            #print(torch.sigmoid(10000.*(m - 0.5)))
            #print(torch.sigmoid(10000.*(m - 0.5)).sum())
            #input("wait")
            num_open_nodes += m.sum()
            #num_open_nodes += torch.sigmoid(10000.*(m-0.5)).sum() # differentiable mask # TODO for some reason this just pushes mask nodes farther apart, not needed anyway as s_factor binarizes masks anyway
            total_mask_size += np.prod(m.size()).item()
        mask_comp = num_open_nodes/total_mask_size
        s_factor = np.prod(self.s_shape)/100
        a_factor = sum(self.a_shape)/4
        #s_factor = np.prod(self.s_shape)/self.max_s_shape
        #a_factor = sum(self.a_shape)/self.max_a_shape
        task_factor = s_factor*a_factor
        #task_factor = np.log((s_factor*a_factor+1) # TODO try? I can't remember which is the right one
        task_comp = task_factor
        #s_factor = np.prod(self.s_shape)
        #a_factor = sum(self.a_shape)
        ##task_factor = np.tanh(self.reg_scaler[self.task_id] * (s_factor*a_factor)) # TODO hyperparam?
        #task_factor = np.tanh(self.reg_scaler_init * (s_factor*a_factor)) # TODO hyperparam?
        # notes:
        # - log7.5 did NOT work
        # - log5 did work
        # - ln did work <- pref this for simplicity and intuitiveness
        #task_comp = np.log10(7.5*task_factor+1)
        #task_comp = np.log10(5*task_factor+1)
        #task_comp = np.log(task_factor+1)
        reg_size = abs(mask_comp - task_comp)
        if 0 == self.task_counters[self.task_id] % 2000:
            print("num_open_nodes:")
            print(num_open_nodes)
            print("total_mask_size:")
            print(total_mask_size)
            print("mask_comp:")
            print(mask_comp)
            print("s_factor:")
            print(s_factor)
            print("a_factor:")
            print(a_factor)
            print("task_factor:")
            print(task_factor)
            print("task_comp:")
            print(task_comp)
            print("reg_size (after mod):")
            print(self.reg_mod*reg_size)
            print("REF - mse_loss:")
            print(mse_loss(input=outputs, target=targets))
        #input("wait")

        return mse_loss(input=outputs, target=targets) + self.lamb * reg_sparse + self.reg_mod * reg_size, (reg_sparse, reg_size)
        #return mse_loss(input=outputs, target=targets) + self.reg_mod * reg_size, reg_size
        #return self.reg_mod * reg_size, reg_size

    def update_target_models(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)

    def act(self, s_sample, is_test):
        a = None

        # with epsilon prob to choose random action else choose argmax Q estimate action
        if (np.random.rand() <= self.epsilon[self.task_id]) and not is_test:
            a = random.randint(0, self.a_shape[0]-1)
        else:
            state = torch.from_numpy(np.expand_dims(s_sample, axis=0)).float()
            state = Variable(state).cuda()

            self.model.eval()
            if is_test:
                estimate, _ = self.model.forward(state, self.smax)
            else:
                estimate, _ = self.model.forward(state, self.s_factor)
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
