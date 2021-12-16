import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.task_ids = []
        self.curr_task_id = -1
        self.task_shapes = {}
        self.fc_s_ins = nn.ModuleList()
        self.fc_a_ins = nn.ModuleList()
        self.fc_outs = nn.ModuleList()
        self.hl_size = 512

        self.fc1 = nn.Linear(self.hl_size, self.hl_size)
        self.efc1 = nn.ModuleList()
        self.fc2 = nn.Linear(self.hl_size, self.hl_size)
        self.efc2 = nn.ModuleList()

        self.relu = nn.ReLU()
        self.gate = nn.Sigmoid()

    def task_configure(self, task_id, s_shape, a_shape):
        if task_id not in self.task_ids:
            if 3 == len(s_shape):
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
                fc_in.append(nn.Linear(2304, int(self.hl_size/2)))
                self.fc_s_ins.append(fc_in)
            else:
                self.fc_s_ins.append(nn.Linear(s_shape[0], int(self.hl_size/2)))
            self.fc_a_ins.append(nn.Linear(a_shape[0], int(self.hl_size/2)))
            self.fc_outs.append(nn.Linear(self.hl_size, 1))

            self.efc1.append(nn.Embedding(1, self.hl_size))
            self.efc2.append(nn.Embedding(1, self.hl_size))

            self.task_ids.append(task_id)
            self.task_shapes[task_id] = len(s_shape)
        self.curr_task_id = task_id

    def forward(self, observations: torch.Tensor, s=1) -> torch.Tensor:
        # gates
        masks = self.mask(self.curr_task_id, s)
        #gfc1, = masks
        gfc1, gfc2 = masks

        x_s, x_a = observations

        #print(x.device)

        if 3 == self.task_shapes[self.curr_task_id]:
            for fc_in in self.fc_s_ins[self.curr_task_id]:
                x_s = fc_in(x_s)
                #print(x.shape)
        else:
            x_s = self.fc_s_ins[self.curr_task_id](x_s)
        x_s = self.relu(x_s)
        #print(x_s.shape)
        x_a = self.fc_a_ins[self.curr_task_id](x_a)
        x_a = self.relu(x_a)
        #print(x_a.shape)
        x = torch.cat([x_s, x_a], 1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        #print(x.shape)
        x = x * gfc1.expand_as(x)
        #print(x.shape)
        x = self.fc2(x)
        x = self.relu(x)
        #print(x.shape)
        x = x * gfc2.expand_as(x)
        #print(x.shape)
        x = self.fc_outs[self.curr_task_id](x)
        #print(x.shape)
        #input("wait")

        return x, masks

    def mask(self, task_id, s=1):
        temp_id = Variable(torch.LongTensor([0]), volatile=False).cuda()
        gfc1 = self.gate(s*self.efc1[self.curr_task_id](temp_id))
        gfc2 = self.gate(s*self.efc2[self.curr_task_id](temp_id))
        #return [gfc1]
        return [gfc1, gfc2]

    def get_view_for(self, n, masks):
        #gfc1, = masks
        gfc1, gfc2 = masks
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
            'efc1': self.efc1,
            'efc2': self.efc2
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
        self.efc2 = checkpoint['efc2']

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.task_ids = []
        self.curr_task_id = -1
        self.task_shapes = {}
        self.fc_ins = nn.ModuleList()
        self.fc_outs = nn.ModuleList()
        self.hl_size = 512

        self.fc1 = nn.Linear(self.hl_size, self.hl_size)
        self.efc1 = nn.ModuleList()
        self.fc2 = nn.Linear(self.hl_size, self.hl_size)
        self.efc2 = nn.ModuleList()

        self.relu = nn.ReLU()
        self.gate = nn.Sigmoid()

    def task_configure(self, task_id, s_shape, a_shape):
        if task_id not in self.task_ids:
            if 3 == len(s_shape):
                fc_in = nn.ModuleList()
                fc_in.append(nn.Conv2d(1, 32, 8, 4))
                fc_in.append(nn.ReLU())
                fc_in.append(nn.Conv2d(32, 64, 4, 2))
                fc_in.append(nn.ReLU())
                fc_in.append(nn.Conv2d(64, 64, 3, 1))
                fc_in.append(nn.ReLU())
                fc_in.append(nn.Flatten())
                fc_in.append(nn.Linear(2304,self.hl_size))
                self.fc_ins.append(fc_in)
            else:
                self.fc_ins.append(nn.Linear(s_shape[0], self.hl_size))
            self.fc_outs.append(nn.Linear(self.hl_size, a_shape[0]))

            self.efc1.append(nn.Embedding(1, self.hl_size))
            self.efc2.append(nn.Embedding(1, self.hl_size))

            self.task_ids.append(task_id)
            self.task_shapes[task_id] = len(s_shape)
        self.curr_task_id = task_id

    def forward(self, observations: torch.Tensor, s=1) -> torch.Tensor:
        # gates
        masks = self.mask(self.curr_task_id, s)
        #gfc1, = masks
        gfc1, gfc2 = masks

        #print(observations.shape)

        x = observations

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
        x = self.fc2(x)
        x = self.relu(x)
        #print(x.shape)
        x = x * gfc2.expand_as(x)
        #print(x.shape)
        x = self.fc_outs[self.curr_task_id](x)
        #print(x.shape)
        #input("wait")

        return x, masks

    def mask(self, task_id, s=1):
        temp_id = Variable(torch.LongTensor([0]), volatile=False).cuda()
        gfc1 = self.gate(s*self.efc1[self.curr_task_id](temp_id))
        gfc2 = self.gate(s*self.efc2[self.curr_task_id](temp_id))
        #return [gfc1]
        return [gfc1, gfc2]

    def get_view_for(self, n, masks):
        #gfc1, = masks
        gfc1, gfc2 = masks
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
            'efc1': self.efc1,
            'efc2': self.efc2
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
        self.efc2 = checkpoint['efc2']
