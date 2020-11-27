from agent import Agent
from env import BTC
from data import retrieve_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDQN on Bitcoin")
    parser.add_argument('-load', type=bool, 
                                default=False)
    parser.add_argument('-games',type=int, default=1000)
    args = parser.parse_args()
    # Load agent, retrieve bitcoin data, and create environment
    load_agent = args.load
    data = retrieve_data()

    env = BTC(data,5000)
    
    
    # Create Agent
    if load_agent:
        print('Trained Agent Loading...')
        agent = Agent(lr=0.00045, input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                    batch_size=16,capacity=1000000 ,fc1_dims=256,fc2_dims=512,eps_dec=5e-7,epsilon=0.1,env=env,replace=1250)

        agent.load()
    else:
        print('Untrained Agent Loading...')
        agent = Agent(lr=0.00045, input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                    batch_size=16,capacity=1000000 ,fc1_dims=256,fc2_dims=512,eps_dec=5e-7,epsilon=1.0,env=env,replace=1250)



    
    scores = []
    running_avg = []
    best_score = -np.inf
    # Agent and Environment 
    print('Starting...')
    for i in trange(args.games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.pick_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            if not load_agent:
                # agent.store_transition(obs, action, reward, obs_, done)
                agent.learn()
            obs = obs_

       # Append scores
        if not load_agent:
            scores.append(score)
        else:
            scores.append(env.total)
        avg_score = np.mean(scores[-100:])
        running_avg.append(avg_score)
        # Saves agent
        if not load_agent:
            if avg_score > best_score:
                # agent.save()
                best_score = avg_score

        # Uncomment to view agent performance
        # print(f'Episode {i}: Score: {score/10000000:.3f} | Avg: {avg_score/10000000:.3f} | Best Score {best_score/10000000:.3f} | Total ${env.total:.2f} |Epsilon {agent.epsilon:.3f} | Reward: {env.reward_dec:.3f}')

    # plt.plot(scores)
    # plt.savefig('scores_a17.png')
    # plt.show()
    plt.title("Agent's Performance over 1000 Episodes (Nov7-Nov25)")
    cm = plt.cm.get_cmap('RdYlBu_r')
    plt.xlabel('Wallet ($)')
    plt.ylabel('Probability')
    # Plot Histogram
    n, bins, patches = plt.hist(running_avg,25,density=True, color='green')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.savefig('Avg_Earnings.png')
    plt.show()


