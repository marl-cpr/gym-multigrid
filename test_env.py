import gym
import time
from gym.envs.registration import register
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='soccer', type=str)

args = parser.parse_args()

def main():

    register(
        id="multigrid-harvest-v0",
        entry_point="gym_multigrid.envs:Harvest4HEnv10x10N2"
    )
    env = gym.make("multigrid-harvest-v0")
    nb_agents = 3


    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        actions = [0] * len(env.agents)
        for i in range(len(env.agents)):
            action = env.agents[i].get_action(obs[0][i], )

        obs, _, done, _ = env.step(agents)

        if done:
            break

if __name__ == "__main__":
    main()
