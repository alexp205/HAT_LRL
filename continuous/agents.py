import random
import numpy as np
import pickle
import math
import copy
import nets
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import mse_loss

class PlayerAgent:
    def __init__(self, gamma, critic_alpha, actor_alpha, seed_val, mem_len, epsilon, epsilon_decay, epsilon_min, smax, c_lamb, a_lamb, thresh_emb, thresh_cosh, clipgrad):
        self.gamma = gamma
        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha
        self.seed_val = seed_val
        self.counter = 0
        self.task_counters = {}
        self.s_shape = (-2)
        self.a_shape = (-2)
        self.task_id = -1
        self.is_first_task = True

        self.s_factor = -1.
        self.critic_mask_pre = None
        self.critic_mask_back = None
        self.actor_mask_pre = None
        self.actor_mask_back = None

        self.critic_model = nets.CriticNet().cuda()
        self.actor_model = nets.ActorNet().cuda()
        self.target_critic_model = nets.CriticNet().cuda()
        self.target_actor_model = nets.ActorNet().cuda()
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())
        self.target_actor_model.load_state_dict(self.actor_model.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_alpha)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_alpha)
        self.memory = {} # set of mems, one per task

        self.mem_len = mem_len
        self.epsilon = {}
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.smax = smax
        self.c_lamb = c_lamb
        self.a_lamb = a_lamb
        self.thresh_emb = thresh_emb
        self.thresh_cosh = thresh_cosh
        self.clipgrad = clipgrad

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

        self.critic_model.task_configure(task_id, s_shape, a_shape)
        if self.target_critic_model.state_dict().keys() != self.critic_model.state_dict().keys():
            print("--- DOING critic PARAM COPY ---")
            temp_critic_model = copy.deepcopy(self.target_critic_model)
            self.target_critic_model = copy.deepcopy(self.critic_model)
            for name, param in self.target_critic_model.named_parameters():
                for t_name, t_param in temp_critic_model.named_parameters():
                    if name == t_name:
                        param.data.copy_(t_param.data)
        self.target_critic_model.task_configure(task_id, s_shape, a_shape)
        self.actor_model.task_configure(task_id, s_shape, a_shape)
        if self.target_actor_model.state_dict().keys() != self.actor_model.state_dict().keys():
            print("--- DOING actor PARAM COPY ---")
            temp_actor_model = copy.deepcopy(self.target_actor_model)
            self.target_actor_model = copy.deepcopy(self.actor_model)
            for name, param in self.target_actor_model.named_parameters():
                for t_name, t_param in temp_actor_model.named_parameters():
                    if name == t_name:
                        param.data.copy_(t_param.data)
        self.target_actor_model.task_configure(task_id, s_shape, a_shape)

        self.task_id = task_id
        if task_id not in self.memory:
            self.memory[task_id] = []
            self.task_counters[task_id] = 0
            self.epsilon[task_id] = self.epsilon_init
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_alpha)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_alpha)

        self.target_critic_model.cuda()
        self.critic_model.cuda()
        self.target_actor_model.cuda()
        self.actor_model.cuda()

    def store_experience(self, s, a, r, s_prime, done, ep, run):
        s_in = s.tolist()
        s_prime_in = s_prime.tolist()

        mem = self.memory[self.task_id]

        if 0 == len(np.asarray(a).shape):
            a = np.reshape(a,(1))

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

    def get_masks(self):
        c_mask = self.critic_model.mask(self.task_id, self.smax)
        for i in range(len(c_mask)):
            c_mask[i] = Variable(c_mask[i].data.clone(), requires_grad=False)
        if self.is_first_task:
            self.critic_mask_pre = c_mask
        else:
            for i in range(len(self.critic_mask_pre)):
                self.critic_mask_pre[i] = torch.max(self.critic_mask_pre[i], c_mask[i])

        self.critic_mask_back = {}
        for n,_ in self.critic_model.named_parameters():
            vals = self.critic_model.get_view_for(n, self.critic_mask_pre)
            if vals is not None:
                self.critic_mask_back[n] = 1 - vals

        #print("critic_mask_pre")
        #print(self.critic_mask_pre)
        #print("critic_mask_back")
        #print(self.critic_mask_back)
        #input("wait")

        a_mask = self.actor_model.mask(self.task_id, self.smax)
        for i in range(len(a_mask)):
            a_mask[i] = Variable(a_mask[i].data.clone(), requires_grad=False)
        if self.is_first_task:
            self.actor_mask_pre = a_mask
        else:
            for i in range(len(self.actor_mask_pre)):
                self.actor_mask_pre[i] = torch.max(self.actor_mask_pre[i], a_mask[i])

        self.actor_mask_back = {}
        for n,_ in self.actor_model.named_parameters():
            vals = self.actor_model.get_view_for(n, self.actor_mask_pre)
            if vals is not None:
                self.actor_mask_back[n] = 1 - vals

        #print("actor_mask_pre")
        #print(self.actor_mask_pre)
        #print("actor_mask_back")
        #print(self.actor_mask_back)
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

        # TODO try including intrinsic for MountainCar

        # ref: self.memory.append((s_in, a, r, s_prime_in, done, ep, run))
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
        self.target_critic_model.eval()
        self.target_actor_model.eval()

        self.s_factor = (self.smax-1/self.smax)*update_num/num_updates+1/self.smax # TODO same for RL? or do we need to modify? 

        a_next, _ = self.target_actor_model.forward(state_new, self.s_factor)
        Q_next, _ = self.target_critic_model.forward([state_new, a_next], self.s_factor)
        y = reward + torch.mul(torch.squeeze(Q_next)*(1-terminal), self.gamma)

        self.critic_optimizer.zero_grad()
        self.critic_model.train()
        Q, c_masks = self.critic_model.forward([state, action], self.s_factor)
        c_loss, _ = self.c_criterion(torch.squeeze(Q), y, c_masks)
        c_loss.backward()

        # HAT - restrict layer grads
        if not self.is_first_task:
            for n,p in self.critic_model.named_parameters():
                if n in self.critic_mask_back:
                    p.grad.data *= self.critic_mask_back[n]
        # HAT - compensate embedding grads
        for n,p in self.critic_model.named_parameters():
            if n.startswith('e'):
                num = torch.cosh(torch.clamp(self.s_factor*p.data, -self.thresh_cosh, self.thresh_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.smax/self.s_factor*num/den
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.clipgrad)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_model.eval()
        self.actor_model.train()
        a_temp, a_masks = self.actor_model.forward(state, self.s_factor)
        Q_temp, _ = self.critic_model.forward([state, a_temp], self.s_factor)
        a_loss, _ = self.a_criterion(Q_temp, a_masks)
        a_loss.backward()

        # HAT - restrict layer grads
        if not self.is_first_task:
            for n,p in self.actor_model.named_parameters():
                if n in self.actor_mask_back:
                    p.grad.data *= self.actor_mask_back[n]
        # HAT - compensate embedding grads
        for n,p in self.actor_model.named_parameters():
            if n.startswith('e'):
                num = torch.cosh(torch.clamp(self.s_factor*p.data, -self.thresh_cosh, self.thresh_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.smax/self.s_factor*num/den
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.clipgrad)
        self.actor_optimizer.step()

        # HAT - constrain embeddings
        for n,p in self.critic_model.named_parameters():
            if n.startswith('e'):
                p.data = torch.clamp(p.data, -self.thresh_emb, self.thresh_emb)
        for n,p in self.actor_model.named_parameters():
            if n.startswith('e'):
                p.data = torch.clamp(p.data, -self.thresh_emb, self.thresh_emb)

        temp_epsilon = self.epsilon[self.task_id]
        temp_epsilon *= self.epsilon_decay
        self.epsilon[self.task_id] = max(self.epsilon_min, temp_epsilon)

    def c_criterion(self, outputs, targets, c_masks):
        c_reg = 0
        c_count = 0
        if self.critic_mask_pre is not None:
            for m,mp in zip(c_masks, self.critic_mask_pre):
                aux = 1-mp
                c_reg += (m*aux).sum()
                c_count += aux.sum()
        else:
            for m in c_masks:
                c_reg += m.sum()
                c_count += np.prod(m.size()).item()
        c_reg /= c_count
        #print("outputs")
        #print(outputs)
        #print("targets")
        #print(targets)
        #print("c_reg")
        #print(c_reg)
        #print("c_count")
        #print(c_count)
        #print("mse_loss:")
        #print(mse_loss(input=outputs, target=targets))
        #input("wait")
        return mse_loss(input=outputs, target=targets) + self.c_lamb * c_reg, c_reg

    def a_criterion(self, outputs, a_masks):
        a_reg = 0
        a_count = 0
        if self.actor_mask_pre is not None:
            for m,mp in zip(a_masks, self.actor_mask_pre):
                aux = 1-mp
                a_reg += (m*aux).sum()
                a_count += aux.sum()
        else:
            for m in a_masks:
                a_reg += m.sum()
                a_count += np.prod(m.size()).item()
        a_reg /= a_count
        #print("outputs")
        #print(outputs)
        #print("targets")
        #print(targets)
        #print("a_reg")
        #print(a_reg)
        #print("a_count")
        #print(a_count)
        #print("mse_loss:")
        #print(mse_loss(input=outputs, target=targets))
        #input("wait")
        return (-outputs).mean() + self.a_lamb * a_reg, a_reg

    def update_target_models(self, tau):
        for target_param, param in zip(self.target_critic_model.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)
        for target_param, param in zip(self.target_actor_model.parameters(), self.actor_model.parameters()):
            target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)

    def act(self, s_sample, is_test):
        a = None

        # with epsilon prob to choose random action else choose argmax Q estimate action
        if (np.random.rand() <= self.epsilon[self.task_id]) and not is_test:
            #a = np.random.rand(self.a_shape[0])
            a = np.random.normal(0, 1, self.a_shape)
        else:
            state = torch.from_numpy(np.expand_dims(s_sample, axis=0)).float()
            state = Variable(state).cuda()

            self.actor_model.eval()
            if is_test:
                estimate, _ = self.actor_model.forward(state, self.smax)
            else:
                estimate, _ = self.actor_model.forward(state, self.s_factor)
            a = estimate[0].data.cpu().numpy()

        #print("check a:")
        #print(a)
        #input("wait")

        return a

    def load(self):
        print("loading model!")
        if self.critic_model:
            self.critic_model.load("./save/critic_model" + str(self.seed_val) + ".h5", self.critic_optimizer)
        if self.target_critic_model:
            self.target_critic_model.load("./save/target_critic_model" + str(self.seed_val) + ".h5", self.critic_optimizer)
        if self.actor_model:
            self.actor_model.load("./save/actor_model" + str(self.seed_val) + ".h5", self.actor_optimizer)
        if self.target_actor_model:
            self.target_actor_model.load("./save/target_actor_model" + str(self.seed_val) + ".h5", self.actor_optimizer)
        #with open("./save/memory.pickle", "rb") as h:
        #    self.memory = pickle.load(h)

    def save(self):
        print("saving model!")
        if self.critic_model:
            self.critic_model.save("./save/critic_model" + str(self.seed_val) + ".h5", self.counter, self.critic_optimizer)
        if self.target_critic_model:
            self.target_critic_model.save("./save/target_critic_model" + str(self.seed_val) + ".h5", self.counter, self.critic_optimizer)
        if self.actor_model:
            self.actor_model.save("./save/actor_model" + str(self.seed_val) + ".h5", self.counter, self.actor_optimizer)
        if self.target_actor_model:
            self.target_actor_model.save("./save/target_actor_model" + str(self.seed_val) + ".h5", self.counter, self.actor_optimizer)
        #with open("./save/memory.pickle", "wb") as h:
        #    pickle.dump(self.memory, h)
