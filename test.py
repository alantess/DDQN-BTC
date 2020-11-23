import unittest
from env import BTC
from data import retrieve_data
from agent import Agent


class MyTestCase(unittest.TestCase):
    # Test action Sampling
    def test_action_sample(self):
        data = retrieve_data()
        env = BTC(data)
        for i in range(10):
            state = env.reset()
            score = 0
            done = False
            while not done:
                action = env.sample_action()
                state_, reward, done, info = env.step(action)
                score += reward
                state = state_
        self.assertTrue(0 <= action <= 8)
# Test Agent's action and Env
    def test_agent(self):
        data = retrieve_data()
        env = BTC(data)
        agent = Agent(lr=0.0003, input_dims=env.observation_space.shape[0],n_actions=env.action_space.shape[0],
                      batch_size=64, epsilon=1.0, env=env)
        for i in range(10):
            state = env.reset()
            done = False
            while not done:
                action = agent.pick_action(state)
                state_, reward, done, info = env.step(action)
                state = state_
                self.assertTrue(0 <= action <= 8)



if __name__ == '__main__':
    unittest.main()
