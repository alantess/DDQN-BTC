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
        self.eps_min = 0.01
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

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.q_eval.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.q_eval.device)
        states_ =T.tensor(states_, dtype=T.float).to(self.q_eval.device)
        done = T.tensor(done, dtype=T.bool).to(self.q_eval.device)

        self.q_train.optimizer.zero_grad()
        self.update_target_network()


        indices = np.arange(self.batch_size)
        q_pred = (self.q_train.forward(states) * actions).sum(dim=1)
        q_next = self.q_eval.forward(states_)
        q_train = self.q_train.forward(states_)

        max_action = T.argmax(q_train,dim=1)
        q_next[done] = 0.0

        y = rewards + self.gamma*q_next[indices,max_action]

        loss = self.q_train.loss(y,q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_train.optimizer.step()



        self.update_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min



