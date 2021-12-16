import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CriticNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(CriticNet, self).__init__()
        self.hl_size = 48

        self.fc1_s = nn.Linear(s_shape[0], int(self.hl_size/2))
        self.fc1_a = nn.Linear(a_shape[0], int(self.hl_size/2))
        self.fc2 = nn.Linear(self.hl_size, self.hl_size)
        self.fc3 = nn.Linear(self.hl_size, 1)

        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x_s, x_a = observations

        x_s = self.fc1_s(x_s)
        x_s = self.relu(x_s)
        x_a = self.fc1_a(x_a)
        x_a = self.relu(x_a)
        x = torch.cat([x_s, x_a], 1)
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

class ActorNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(ActorNet, self).__init__()
        self.hl_size = 48

        self.fc1 = nn.Linear(s_shape[0], self.hl_size)
        self.fc2 = nn.Linear(self.hl_size, self.hl_size)
        self.fc3 = nn.Linear(self.hl_size, a_shape[0])

        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        x = self.fc1(x)
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

class TauModelNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(TauModelNet, self).__init__()
        self.hl_size = 48

        self.fc_s_in = nn.Linear(s_shape[0], int(self.hl_size/2))
        self.fc_a_in = nn.Linear(a_shape[0], int(self.hl_size/2))
        self.fc1 = nn.Linear(self.hl_size, self.hl_size)
        self.fc_out = nn.Linear(self.hl_size, s_shape[0])

        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x_s, x_a = observations

        x_s = self.fc_s_in(x_s)
        x_s = self.relu(x_s)
        x_a = self.fc_a_in(x_a)
        x_a = self.relu(x_a)
        x = torch.cat([x_s, x_a], 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_out(x)

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
