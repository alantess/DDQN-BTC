import unittest
from env import BTC
from data import retrieve_data


class MyTestCase(unittest.TestCase):
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



if __name__ == '__main__':
    unittest.main()
