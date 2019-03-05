# 2-agent Guided-Navigation environment without moving obstacles.
# ===============================================================

import random
import sys

import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple, Box
from six import StringIO


class GuidedNavigation(Env):
    def __init__(self):
        """Initialization method for the environment Guided-Navigation

        """
        self.noise = 0.1  # (executed actions) != (intended actions) with this probability
        self.max_y = 10
        self.max_x = 8
        self.footpath_blocks = [1, 1, 0, 0, 1, 1, 1, 1]
        self.footpath_position = [2, 7]
        self.trees = [(5, 0), (1, 9)]
        self.poles = [(5, 9), (1, 1)]
        self.static_map = {}
        for coordinate_y in range(self.max_y):
            if coordinate_y not in self.static_map:
                self.static_map[coordinate_y] = [0] * self.max_x
            for coordinate_x in range(self.max_x):
                if coordinate_y in self.footpath_position and \
                        self.footpath_blocks[coordinate_x] == 1:
                    self.static_map[coordinate_y][coordinate_x] = 'X'
                else:
                    self.static_map[coordinate_y][coordinate_x] = ' '
        for item in self.trees:
            self.static_map[item[1]][item[0]] = 'T'
        for item in self.poles:
            self.static_map[item[1]][item[0]] = 'P'

        self.goal = [(6, 9), (7, 9)]
        self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        self.action_space = Tuple(
            [Discrete(len(self.actions)), Discrete(len(self.actions))]
        )
        self.reset()
        self.low_agent1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.high_agent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.low_agent2 = np.array([0, 0])
        self.high_agent2 = np.array([self.max_x, self.max_y])
        self.low = (self.low_agent1, self.low_agent2)
        self.high = (self.high_agent1, self.high_agent2)
        self.observation_space_agent1 = Box(self.low_agent1, self.high_agent1, dtype=np.int)
        self.observation_space_agent2 = Box(self.low_agent2, self.high_agent2, dtype=np.int)
        self.observation_space = Tuple([self.observation_space_agent1,
                                        self.observation_space_agent2])

    def reset(self):
        """method to reset the environment to the initial state

        :return: observations of the environment
        """

        """ Dog observations - a list of binary values for 10 positions as mentioned below
            ---------
            |2|3|4|5|
            |1|D|M|6|
            |0|9|8|7|
            ---------
            0 - no obstacle
            1 - obstacle
        """

        self.state = (7, 0)
        obstacles = self._detect_obstacles((7, 0))
        return self._get_obs(obstacles, (7, 0))

    def _get_obs(self, obstacles, position):
        """method to get the observations of the agents; returns a numpy array

        :param obstacles: obstacles around the agents
        :param position: agent position
        :return: np array of the agents' state
        """
        return np.array((obstacles, position))

    def render(self, mode='human'):
        """Render method for the environment; overridden the parent render method

        :param mode: human/ansi; used to determine the output format or console
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        self._print_state(outfile)

    def _print_state(self, output_stream):
        """print method to print the state of the environment

        :param output_stream: output console
        """
        agent2_state = self.state
        state = ''
        for j in range(self.max_y - 1, -1, -1):
            for i in range(self.max_x):
                if i == int(agent2_state[0]) - 1 and j == int(agent2_state[1]):
                    state += 'D'  # agent 1 - Dog
                elif i == int(agent2_state[0]) and j == int(agent2_state[1]):
                    state += 'M'  # agent 2 - Blind Man
                else:
                    state += self.static_map[j][i]
            state += '\n'
        output_stream.write(state)

    def _move_agent_horizontal(self, agent_position, obstacles, action):
        """function for the horizontal movement of the agents

        :param agent_position: position of the agents
        :param obstacles: obstacles around the obstacle
        :param action: action of the agent
        :return agent_position: updated when no obstacles else the same
        """
        a_x, a_y, dx = agent_position[0], agent_position[1], -1 if action == 0 else 1 if action == 1 else 0
        nx = a_x + dx
        if (nx < 0) or (nx >= self.max_x) or self.static_map[a_y][nx] != ' ' or (a_x - 1 + dx) < 0 or \
                (a_x - 1 + dx) >= self.max_x or self.static_map[a_y][a_x - 1 + dx] != ' ' or (
                dx == -1 and obstacles[1] == 1) or (dx == 1 and obstacles[6] == 1):
            return agent_position
        agent_position[0] = nx
        return agent_position

    def _move_agent_vertical(self, agent_position, obstacles, action):
        """function for the vertical movement of the agents

        :param agent_position: position of the agents
        :param obstacles: obstacles around the obstacle
        :param action: action of the agent
        :return agent_position: updated when no obstacles else the same
        """
        a_x, a_y, dy = agent_position[0], agent_position[1], -1 if action == 3 else 1 if action == 2 else 0
        ny = a_y + dy
        if (ny < 0) or (ny >= self.max_y) or (a_x - 1) < 0 or self.static_map[ny][a_x - 1] != ' ' or \
                self.static_map[ny][a_x] != ' ' or (
                dy == -1 and (obstacles[8] == 1 or obstacles[9] == 1)) or \
                (dy == 1 and (obstacles[3] == 1 or obstacles[4] == 1)):
            return agent_position
        agent_position[1] = ny
        return agent_position

    def _detect_obstacles(self, agent_position):
        """function to detect obstacles after the action of the agents is performed

        :param agent_position: position of the agents
        :return state of agent - dog (obstacles around the agents)
        """

        obstacles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        surroundings = {"0": (agent_position[0] - 2, agent_position[1] - 1),
                        "1": (agent_position[0] - 2, agent_position[1]),
                        "2": (agent_position[0] - 2, agent_position[1] + 1),
                        "3": (agent_position[0] - 1, agent_position[1] + 1),
                        "4": (agent_position[0], agent_position[1] + 1),
                        "5": (agent_position[0] + 1, agent_position[1] + 1),
                        "6": (agent_position[0] + 1, agent_position[1]),
                        "7": (agent_position[0] + 1, agent_position[1] - 1),
                        "8": (agent_position[0], agent_position[1] - 1),
                        "9": (agent_position[0] - 1, agent_position[1] - 1)}
        for key, value in surroundings.items():
            if value[1] < 0 or value[1] >= self.max_y \
                    or value[0] < 0 or value[0] >= self.max_x \
                    or self.static_map[value[1]][value[0]] != ' ':
                obstacles[int(key)] = 1
            else:
                obstacles[int(key)] = 0
        return obstacles

    def is_goal(self):
        if (self.state[0], self.state[1]) in self.goal \
                and (self.state[0] - 1, self.state[1]) in self.goal:
            return True
        else:
            return False

    def step(self, actions):
        """

        :param actions: actions to be performed by the agent
        :return: tuple
            observation: final observation state of the agents after the action
            reward: reward for the current action
            done: true if goal reached; else false
            info: any information if needed
        """
        action1, action2 = int(actions[0]), int(actions[1])
        # print('actual actions: ', actions)
        if self.noise > 0:
            x = random.random()
            if x < self.noise:
                random_action = random.randrange(len(self.actions) - 1)
                action1, action2 = random_action, random_action
                # print('noise actions:', action1, action2)

        agent_position = list(self.state)

        obstacles = self._detect_obstacles(agent_position)
        if action1 in (1, 0):
            agent_position = self._move_agent_horizontal(agent_position, obstacles, action1)
        elif action1 in (2, 3):
            agent_position = self._move_agent_vertical(agent_position, obstacles, action1)
        else:
            pass

        obstacles = self._detect_obstacles(agent_position)
        self.state = tuple(agent_position)

        done, reward = False, 0

        if self.is_goal():
            done = True
            reward = 100
        else:
            reward = -1

        return self._get_obs(tuple(obstacles), tuple(agent_position)), reward, done, {}

    # def _action_values(self, actions):
    #     """
    #
    #     :param actions: list of actions
    #     :return: list of action numbers
    #     """
    #     return list(actions.index(x) for x in actions)
    #
    # def _env_name(self):
    #     """
    #
    #     :return: domain name
    #     """
    #     return "GuideDog-v0"


# GUIDE_DOG = GuidedNavigation()
# GUIDE_DOG.render()
# reward = 0
# steps = 0
# print('initial state: ', GUIDE_DOG.state)
# while True:
#     action1 = int(input("Agent 1 action"))
#     action2 = int(input("Agent 2 action"))
#     obs, rew, done, info = GUIDE_DOG.step((action1, action2))
#     GUIDE_DOG.render()
#     print('after step: ' + str(GUIDE_DOG.state))
#     steps += 1
#     reward += rew
#     print('Reward: ', reward)
#     if done:
#         print('Goal Reached, total reward is ' + str(reward))
#         print('steps: ', steps)
#         break
