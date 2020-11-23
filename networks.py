import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os


class DDQN(nn.Module):
    def __init__(self, lr, input_dims, n_actions,fc1_dims, fc2_dims,network_name, chkpt_dir="models"):
        super(DDQN, self).__init__()
        self.file = os.path.join(chkpt_dir, network_name+'_ddqn')
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, n_actions)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr =lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q = self.q(x)
        return q

    def save(self):
        T.save(self.load_state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))
