import numpy as np
import torch as T
import torch.nn as nn
from networks import DDQN
from memory import ReplayBuffer

class Agent(object):
    def __init__(self, lr, input_dims, n_actions,epsilon, batch_size,env,
                 capacity=1000000, eps_dec=5e-7, fc1_dims = 256, fc2_dims=256,
                 repalce=1000, gamma=0.99,):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.memory = ReplayBuffer(capacity, input_dims,n_actions)
        self.eps_dec = eps_dec
        self.replace = repalce
        self.update_cntr = 0

        # Evaluate network
        self.q_eval = DDQN(lr=lr, input_dims=self.input_dims,n_actions=self.n_actions,fc1_dims=fc1_dims, fc2_dims=fc2_dims,network_name='_eval')
        # Training Network
        self.q_train = DDQN(lr=lr, input_dims=self.input_dims,n_actions=self.n_actions,fc1_dims=fc1_dims, fc2_dims=fc2_dims,network_name='_train')

    def pick_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor([obs], dtype=T.float).to(self.q_eval.device)
            actions = self.q_train.forward(state)
            action = T.argmax(actions).item()
        else:
            action = self.env.sample_action()

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward,state_,done)


    def update_target_network(self):
        if self.update_cntr % self.replace == 0:
            self.q_eval.load_state_dict(self.q_train.state_dict())

    def save(self):
        print('Saving...')
        self.q_eval.save()
        self.q_train.save()

    def load(self):
        print('Loading...')
        self.q_eval.load()
        self.q_train.load()