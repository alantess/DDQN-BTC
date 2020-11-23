import numpy as np
from gym.utils import seeding

class BTC(object):
    def __init__(self, data, investment=5000):
        self.data = data
        self.n_steps, self.n_headers = data.shape
        self.time_step = 0
        self.initial_investment = investment
        self.usd_wallet = None
        self.reward_dec = 1.0 
        self.btc_wallet = None
        self.btc_price = None
        self.total = 0
        self.profits = [] 
        # Action Space Holds 9 different Action from Hold to (25% - 100%) Buy
        self.action_space = np.arange(9)
        # Vector [ close, high, low, open, bitcoin wallet, usd wallet]
        self.observation_space = np.empty(self.n_headers + 2, dtype=np.float)
        self.reset()
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.time_step = 0
        self.profits= []
        self.total = 0
        self.btc_wallet = 0
        self.usd_wallet = self.initial_investment
        self._get_price()
        self.reward_dec = self.reward_dec - 0.99e-3 if self.reward_dec > 0 else 0 
        return self._get_state()

    def step(self,action):
        assert action in self.action_space
        # Retrieve price
        self._get_price()
        reward = 0.0
        # Holds from previous state
        prev_holdings = self.btc_wallet + self.usd_wallet
        # Select an action, update wallets and increment time step
        self._action_set(action)
        self._update_btc_wallet()
        self.time_step += 1
        # Holdings from new state
        new_holdings = self.btc_wallet + self.usd_wallet
        self.total = new_holdings
        self.profits.append(new_holdings) 
        # Determines whether the trade was good or bad
        earnings_ratio  = new_holdings / prev_holdings

       # Get s', check if done, and info on the state
        state_ = self._get_state()

        done = self.time_step == self.n_steps - 1

        info = {'Bitcoin': self.btc_wallet,
                'USD': self.usd_wallet}

        rewards = (new_holdings - prev_holdings) * self.reward_dec
        reward = (rewards * 0.5) + (np.linalg.norm(self.profits) * 0.01) 
        if done:
            if self.total > 70 * self.initial_investment:
                reward += 100
            else:
                reward -=100
        # Combined  Reward
        # if earnings_ratio == 1:
            # reward = 0.0
        # elif earnings_ratio > 1:
            # reward = (self.reward_dec * 0.02) + 0.1 
        # else:
            # reward = (self.reward_dec * -0.02) - 0.1

        # Implement Reward Delay using timestep , total holdings, and investment  
#         if self.time_step % 30 == 0:
            # if self.total > self.initial_investment:
                # reward += 0.05
            # else:
                # reward - 0.05

        if done:
            if new_holdings > 80 * self.initial_investment:
                reward = 1
            else:
                reward = -1

        return state_, reward, done, info
    # Sample a random action from action space
    def sample_action(self):
        action = np.random.choice(self.action_space)
        return action


    # Make a Trade or Hold
    def _action_set(self,action):
        # Actions correspond with selling or buying bitcoin
        # Hold
        if action == 0:
            return
        # Purchase 100%
        if action == 1:
            self._buy_or_sell(purchase=True, percentage=1.0)
        # Sell 100%
        if action == 2:
            self._buy_or_sell(purchase=False,percentage=1.0)
        # Purchase 75%
        if action == 3:
            self._buy_or_sell(purchase=True,percentage=0.75)
        # Sell 75%
        if action == 4:
            self._buy_or_sell(purchase=False, percentage=0.75)
        # Purchase 50%
        if action == 5:
            self._buy_or_sell(purchase=True,percentage=0.5)
        # Sell 50%
        if action == 6:
            self._buy_or_sell(purchase=False,percentage=0.5)
        # Purchase 25%
        if action == 7:
            self._buy_or_sell(purchase=True, percentage=0.25)
        # Sell 25%
        if action == 8:
            self._buy_or_sell(purchase=False, percentage=0.25)

    def _buy_or_sell(self, purchase, percentage):
        #  Purchase or Sell Amount
        amount = self.btc_price * percentage
        if purchase:
            if self.usd_wallet > amount:
                self.usd_wallet -= amount
                self.btc_wallet += amount
        else:
            if self.btc_wallet >= amount:
                self.btc_wallet -= amount
                self.usd_wallet += amount


    def _update_btc_wallet(self):
        self.btc_wallet *= self.data[self.time_step+1][0] / self.btc_price

    def _get_state(self):
        state = self.observation_space
        state[:4] = self.data[self.time_step]
        state[4] = self.btc_wallet
        state[5] = self.usd_wallet
        return  state

    def _get_price(self):
        self.btc_price = self.data[self.time_step][0]






