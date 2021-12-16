import random
import numpy as np
import pickle
import math
import copy
import CHECK_nets as nets
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import mse_loss

from sklearn.mixture import BayesianGaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from sumtree import SumTree

# ref: https://github.com/rlcode/per
class PER():
    def __init__(self, cap):
        self.tree = SumTree(cap)
        self.cap = cap

        self.e = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_inc = 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total()/n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_inc])
        for i in range(n):
            a = segment*i
            b = segment*(i+1)

            s = random.uniform(a,b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sample_probs = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sample_probs, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class PlayerAgent:
    def __init__(self, gamma, critic_alpha, actor_alpha, tau_alpha, seed_val, mem_len, epsilon, epsilon_decay, epsilon_min, s_shape, a_shape, i_mod):
        self.gamma = gamma
        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha
        self.tau_alpha = tau_alpha
        self.seed_val = seed_val
        self.counter = 0
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.critic_model = nets.CriticNet(self.s_shape, self.a_shape).cuda()
        self.actor_model = nets.ActorNet(self.s_shape, self.a_shape).cuda()
        self.target_critic_model = nets.CriticNet(self.s_shape, self.a_shape).cuda()
        self.target_actor_model = nets.ActorNet(self.s_shape, self.a_shape).cuda()
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())
        self.target_actor_model.load_state_dict(self.actor_model.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_alpha)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_alpha)
        self.memory = []
        #self.memory = PER(mem_len)

        ## ref: https://scikit-learn.org/stable/modules/mixture.html#bgmm
        ## ref: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
        #self.n_comps = 2 # TODO hyperparam...
        #self.dens_model = BayesianGaussianMixture(n_components=self.n_comps, covariance_type='diag', random_state=self.seed_val, warm_start=False, max_iter=1000, reg_covar=1e-4, tol=1e-3, weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None) # TODO hyperparam...
        ## hyperparam notes: (covariance_type: {full, diag}, weight_concentration_prior_type: {dirichlet_process, dirichlet_distribution}, weight_concentration_prior: {None, some num})
        #self.dens_model.fit(np.random.rand(self.n_comps,*s_shape)) # need to do this b/c sklearn is weird
        #self.c = 0.1
        #self.tau_model = nets.TauModelNet(self.s_shape, self.a_shape).cuda()
        #self.tau_optimizer = optim.Adam(self.tau_model.parameters(), lr=self.tau_alpha)
        #self.i_mod = i_mod

        self.mem_len = mem_len
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def set_up_task(self, s_shape, a_shape):
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.target_critic_model.cuda()
        self.critic_model.cuda()
        self.target_actor_model.cuda()
        self.actor_model.cuda()
        #self.tau_model.cuda()

    def store_experience(self, s, a, r, s_prime, done, ep, run, rho, rho_updated):
        s_in = s.tolist()
        s_prime_in = s_prime.tolist()

        mem = self.memory

        if 0 == len(np.asarray(a).shape):
            a = np.reshape(a,(1))

        if len(mem) < self.mem_len:
            mem.append((s_in, a, r, s_prime_in, done, ep, run, rho, rho_updated))
        else:
            #replace_idx = random.randrange(len(mem)) # rand
            replace_idx = self.counter % self.mem_len # wrap
            mem[replace_idx] = (s_in, a, r, s_prime_in, done, ep, run, rho, rho_updated)
        self.counter += 1
        #self.critic_model.eval()
        #target = self.critic_model.forward([Variable(torch.from_numpy(np.expand_dims(s, axis=0)).float()).cuda(), Variable(torch.from_numpy(np.expand_dims(a, axis=0)).float()).cuda()])
        ##print("target:")
        ##print(target)
        #val = target.data.cpu().numpy()[0][0]
        ##print("val:")
        ##print(val)
        #a_prime = self.actor_model.forward(Variable(torch.from_numpy(np.expand_dims(s_prime, axis=0)).float()).cuda())
        #target_prime = self.critic_model.forward([Variable(torch.from_numpy(np.expand_dims(s_prime, axis=0)).float()).cuda(), a_prime])
        ##print("a_prime:")
        ##print(a_prime)
        ##print("target_prime:")
        ##print(target_prime)
        #val_prime = (r + torch.mul(torch.squeeze(target_prime)*(1-done), self.gamma)).data.cpu().numpy()
        ##print("val_prime:")
        ##print(val_prime)
        #error = abs(val - val_prime)
        ##print("error:")
        ##print(error)
        #mem.add(error, (s_in, a, r, s_prime_in, done, ep, run, rho, rho_updated))
        ##input("wait")

    def rescale(self, val, in_min, in_max, out_min, out_max):
        return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

    def sigmoid(self, val):
        return 1. / (1. + np.exp(-val))

    # ref: https://arxiv.org/pdf/1902.08039.pdf
    # TODO check
    def update_dens_models(self, s):
        s = s.reshape(1,-1)

        # fix for no warm_start and no partial_fit
        if self.counter < self.n_comps:
            state = np.random.rand(self.n_comps, *self.s_shape)
        else:
            #mem = np.asarray(self.memory, dtype=object)
            mem = np.asarray(self.memory, dtype=object)[max(0, min(self.counter, self.mem_len)-20000):]
            #mem = np.asarray(self.memory.tree.data[:self.memory.tree.n_entries].tolist(), dtype=object)
            state = np.asarray(mem[:,0].tolist())
        #print(state.shape)
        self.dens_model.fit(state)

        rho = self.dens_model.predict_proba(s)
        #print("densities:")
        #print(rho)
        w = self.dens_model.weights_
        #print("w:")
        #print(w)
        rho = np.dot(rho, w)[0]
        #print("rho:")
        #print(rho)
        state = np.append(state, s, axis=0)
        #print(state.shape)
        #self.dens_model.fit(np.tile(s,(self.n_comps,1)))
        self.dens_model.fit(state)
        rho_updated = self.dens_model.predict_proba(s)
        #print("updated densities:")
        #print(rho_updated)
        w_updated = self.dens_model.weights_
        #print("w_updated:")
        #print(w_updated)
        rho_updated = np.dot(rho_updated, w_updated)[0]
        #print("rho_updated:")
        #print(rho_updated)
        #input("wait")

        return (rho, rho_updated)

    def update_models(self, batch_size):
        record_range = min(self.counter, self.mem_len)
        batch_indices = np.random.choice(record_range, batch_size)
        batch = np.asarray(self.memory, dtype=object)[batch_indices]
        #batch, idxs, is_weights = self.memory.sample(batch_size)
        #batch = np.asarray(batch, dtype=object)

        reward = torch.from_numpy(np.asarray(batch[:,2].tolist())).float()
        #r_ei = []
        ### ref: https://lilianweng.github.io/lil-log/2020/06/07/exploration-strategies-in-deep-reinforcement-learning.html#count-based-exploration
        ### ref: https://arxiv.org/pdf/1703.01310.pdf
        ### note: c = 0.1 by default, n is curr iter
        ##for s, a, r, s_prime, done, ep, run, rho, rho_updated in batch:
        ##    PG = max(np.log(rho_updated) - np.log(rho), 0.)
        ##    N = 1./((np.exp(self.c * np.power(self.counter, -1./2.) * PG) - 1.) + 1e-10)
        ##    r_i = np.power(N, -1./2.)
        ##    r_mod = r + self.i_mod * r_i
        ##    #if 0 == self.counter % 2000:
        ##    #    print("r:")
        ##    #    print(r)
        ##    #    print("PG:")
        ##    #    print(PG)
        ##    #    print("N:")
        ##    #    print(N)
        ##    #    print("r_i:")
        ##    #    print(r_i)
        ##    #    print("r_mod:")
        ##    #    print(r_mod)
        ##    r_ei.append(r_mod)
        #tau_model_copy = nets.TauModelNet(self.s_shape, self.a_shape).cuda()
        #for s, a, r, s_prime, done, ep, run, rho, rho_updated in batch:
        #    tau_model_copy.load_state_dict(self.tau_model.state_dict())
        #    tau_optimizer_copy = optim.Adam(tau_model_copy.parameters(), lr=self.tau_alpha)

        #    s_temp = torch.from_numpy(np.expand_dims(s, axis=0)).float()
        #    s_temp = Variable(s_temp).cuda()
        #    a_temp = torch.from_numpy(np.expand_dims(a, axis=0)).float()
        #    a_temp = Variable(a_temp).cuda()
        #    s_prime_temp = torch.from_numpy(np.expand_dims(s_prime, axis=0)).float()
        #    s_prime_temp = Variable(s_prime_temp).cuda()

        #    tau_model_copy.train()
        #    temp_loss = mse_loss(input=tau_model_copy.forward([s_temp, a_temp]), target=s_prime_temp.detach())
        #    tau_optimizer_copy.zero_grad()
        #    temp_loss.backward()
        #    tau_optimizer_copy.step()
        #    tau_model_copy.eval()
        #    temp_loss_updated = mse_loss(input=tau_model_copy.forward([s_temp, a_temp]), target=s_prime_temp.detach())
        #    r_i = max(temp_loss.data.cpu().numpy() - temp_loss_updated.data.cpu().numpy(), 0.)
        #    r_mod = r + self.i_mod * r_i
        #    #if 0 == self.counter % 2000:
        #    #    print("temp_loss:")
        #    #    print(temp_loss)
        #    #    print("temp_loss_updated:")
        #    #    print(temp_loss_updated)
        #    #    print("r_i:")
        #    #    print(r_i)
        #    #    print("r_mod:")
        #    #    print(r_mod)
        #    r_ei.append(r_mod)
        ##input("wait")
        #reward = torch.from_numpy(np.asarray(r_ei)).float()

        # ref: self.memory.append((s_in, a, r, s_prime_in, done, ep, run))
        state = torch.from_numpy(np.asarray(batch[:,0].tolist())).float()
        action = torch.from_numpy(np.asarray(batch[:,1].tolist())).float()
        state_new = torch.from_numpy(np.asarray(batch[:,3].tolist())).float()
        terminal = torch.from_numpy(np.asarray(batch[:,4].tolist())).float()
        state = Variable(state).cuda()
        action = Variable(action).cuda()
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()
        self.target_critic_model.eval()
        self.target_actor_model.eval()

        a_next = self.target_actor_model.forward(state_new)
        Q_next = self.target_critic_model.forward([state_new, a_next])
        y = reward + torch.mul(torch.squeeze(Q_next)*(1-terminal), self.gamma)

        self.critic_optimizer.zero_grad()
        self.critic_model.train()
        Q = self.critic_model.forward([state, action])
        c_loss = self.c_criterion(torch.squeeze(Q), y)
        c_loss.backward()

        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_model.eval()
        self.actor_model.train()
        a_temp = self.actor_model.forward(state)
        Q_temp = self.critic_model.forward([state, a_temp])
        a_loss = self.a_criterion(Q_temp)
        a_loss.backward()

        self.actor_optimizer.step()

        #action = torch.from_numpy(np.asarray(batch[:,1].tolist())).float()
        #action = Variable(action).cuda()
        #self.tau_model.train()
        #s_prime_pred = self.tau_model.forward([state, action])
        #tau_loss = mse_loss(input=s_prime_pred, target=state_new.detach())
        #self.tau_optimizer.zero_grad()
        #tau_loss.backward()
        #self.tau_optimizer.step()

        temp_epsilon = self.epsilon
        temp_epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, temp_epsilon)

    def c_criterion(self, outputs, targets):
        return mse_loss(input=outputs, target=targets)

    def a_criterion(self, outputs):
        return (-outputs).mean()

    def update_target_models(self, tau):
        for target_param, param in zip(self.target_critic_model.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)
        for target_param, param in zip(self.target_actor_model.parameters(), self.actor_model.parameters()):
            target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)

    def act(self, s_sample, is_test):
        a = None

        # with epsilon prob to choose random action else choose argmax Q estimate action
        if (np.random.rand() <= self.epsilon) and not is_test:
            #a = np.random.rand(self.a_shape[0])
            a = np.random.normal(0, 1, self.a_shape)
        else:
            state = torch.from_numpy(np.expand_dims(s_sample, axis=0)).float()
            state = Variable(state).cuda()

            self.actor_model.eval()
            estimate = self.actor_model.forward(state)
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
