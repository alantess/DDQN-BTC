from agent import Agent
from env import BTC
from data import retrieve_data
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = retrieve_data()
    env = BTC(data)
    agent = Agent(lr=0.0003, input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                  batch_size=256, epsilon=1.0, env=env, replace=3000)
    scores = []
    running_avg = []
    best_score = -np.inf
    for i in range(1000):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.pick_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            obs = obs_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        running_avg.append(avg_score)
        if avg_score > best_score:
            agent.save()
            best_score = avg_score

        print(f'Episode {i}:\tScore {score:.3f} | Average Score {avg_score:.3f} | Best Score {best_score:.2f} | {env.total:.2f} |Epsilon {agent.epsilon:.3f} | Reward: {env.reward_dec:.3f}')


    plt.plot(running_avg)
    plt.savefig('score_plt.png')


