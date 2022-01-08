import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from envs.mdp import StochasticMDP
from agent.hDQN import hDQN

plt.style.use('ggplot')

def one_hot(state):
    vector = np.zeros(6)
    vector[state] = 1.0
    return np.expand_dims(vector, axis=0)

def main():
    ActorExperience = namedtuple('ActorExperience', ['state', 'goal', 'action', 'reward', 'next_state', 'done'])
    MetaExperience = namedtuple('MetaExperience', ['state', 'goal', 'reward', 'next_state', 'done'])

    env = StochasticMDP()
    agent = hDQN()

    n_states = 6
    n_episodes = 12000
    episode_div_factor = 1000
    visits = np.zeros((n_episodes // episode_div_factor, n_states))
    anneal_factor = 0.9 / n_episodes
    print('Annealing factor: ' + str(anneal_factor))
    for episode in range(n_episodes):
        episode_thousand = episode // episode_div_factor

        print('\n\n### EPISODE '  + str(episode) + '###')
        state = env.reset()
        visits[episode_thousand][state] += 1
        done = False
        while not done:
            goal = agent.select_goal(one_hot(state))
            agent.goal_selected[goal] += 1
            print('\nNew Goal: '  + str(goal) + '\nState-Actions: ')
            total_external_reward = 0
            goal_reached = False
            while not done and not goal_reached:
                action = agent.select_move(one_hot(state), one_hot(goal), goal)
                print(str((state,action)) + '; ')
                next_state, external_reward, done = env.step(action)
                visits[episode_thousand][next_state] += 1
                intrinsic_reward = agent.criticize(goal, next_state)
                goal_reached = next_state == goal
                if goal_reached:
                    agent.goal_success[goal] += 1
                    print('Goal reached!! ')
                if next_state == 5:
                    print('S5 reached!! ')
                exp = ActorExperience(one_hot(state), one_hot(goal), action, intrinsic_reward, one_hot(next_state), done)
                agent.store(exp, meta=False)
                agent.update(meta=False)
                agent.update(meta=True)
                total_external_reward += external_reward
                state = next_state
            exp = MetaExperience(one_hot(state), one_hot(goal), total_external_reward, one_hot(next_state), done)
            agent.store(exp, meta=True)
            
            #Annealing 
            agent.meta_epsilon -= anneal_factor
            avg_success_rate = agent.goal_success[goal] / agent.goal_selected[goal]
            
            if(avg_success_rate == 0 or avg_success_rate == 1):
                agent.actor_epsilon -= anneal_factor
            else:
                agent.actor_epsilon = 1- avg_success_rate
        
            if(agent.actor_epsilon < 0.1):
                agent.actor_epsilon = 0.1
            print('meta_epsilon: ' + str(agent.meta_epsilon))
            print('actor_epsilon ' + str(goal) + ': ' + str(agent.actor_epsilon))
            
        if (episode % 100 == 99):
            print('')
            print(str(visits/1000) + '')

    eps = list(range(1,13))
    plt.subplot(2, 3, 1)
    plt.plot(eps, visits[:,0]/1000)
    plt.xlabel('Episodes (*1000)')
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title('S1')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(eps, visits[:,1]/1000)
    plt.xlabel('Episodes (*1000)')
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title('S2')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(eps, visits[:,2]/1000)
    plt.xlabel('Episodes (*1000)')
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title('S3')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(eps, visits[:,3]/1000)
    plt.xlabel('Episodes (*1000)')
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title('S4')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(eps, visits[:,4]/1000)
    plt.xlabel('Episodes (*1000)')
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title('S5')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(eps, visits[:,5]/1000)
    plt.xlabel('Episodes (*1000)')
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title('S6')
    plt.grid(True)
    plt.savefig('first_run.png')
    plt.show()

if __name__ == '__main__':
    main()
