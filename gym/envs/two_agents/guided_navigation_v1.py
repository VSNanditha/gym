# 2-agent guide dog environment.
# ==============================

import random
import sys

import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple, Box
from six import StringIO


class GuidedNavigation(Env):
    def __init__(self):
        """Initialization method for the environment Guide-Dog

        """
        self.noise = 0.1  # (executed actions) != (intended actions) with this probability
        self.max_y = 10
        self.max_x = 8
        self.footpath_blocks = [1, 1, 0, 0, 1, 1, 1, 1]
        self.footpath_position = [2, 7]
        self.trees = [(5, 0), (1, 9)]
        self.poles = [(5, 9), (1, 1)]
        self.moving_objects = {}
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
        self.actions_agent1 = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'SEND-TUG', 'SEND-PUSH', 'SEND-DOWN']
        self.actions_agent2 = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        self.action_space = Tuple(
            [Discrete(len(self.actions_agent1)), Discrete(len(self.actions_agent2))]
        )
        self.horizontal_objects = 0
        self.vertical_objects = 0
        self.reset()
        self.low_agent1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.high_agent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3])
        self.low_agent2 = np.array([0, 0, 0])
        self.high_agent2 = np.array([self.max_x, self.max_y, 3])
        self.low = (self.low_agent1, self.low_agent2)
        self.high = (self.high_agent1, self.high_agent2)
        self.observation_space_agent1 = Box(self.low_agent1, self.high_agent1, dtype=np.int)
        self.observation_space_agent2 = Box(self.low_agent2, self.high_agent2, dtype=np.int)
        self.observation_space = Tuple([self.observation_space_agent1,
                                        self.observation_space_agent2])

    def _get_moving_objects(self, horizontal_objects=0, vertical_objects=0):
        """function to set the number of moving objects in the domain

        :param horizontal_objects:
        :param vertical_objects:
        """
        self.horizontal_objects = horizontal_objects
        self.vertical_objects = vertical_objects

    def reset(self):
        """method to reset the environment to the initial state

        :param horizontal_objects: number of moving objects for the environment
                                    in horizontal direction
        :param vertical_objects: number of moving objects for the environment
                                    in vertical direction
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
        assert self.horizontal_objects in range(0, 3), "Error in the number of horizontal moving " \
                                                       "objects, allowed max 2 min 0"
        assert self.vertical_objects in range(0, 4), "Error in the number of vertical moving " \
                                                     "objects, allowed max 3 min 0"

        starting_position = [(0, 4), (0, 5), (0, 6), (7, 4), (7, 5), (7, 6)]
        for obj in range(self.horizontal_objects):
            random_position = random.sample(starting_position, 1)
            key = str(obj) + '_h'
            self.moving_objects[key] = ['h', random_position[0]]
            starting_position.remove(random_position[0])

        starting_position = (3, 9)
        for obj in range(self.vertical_objects):
            key = str(obj) + '_v'
            self.moving_objects[key] = ['v', starting_position]

        self.state = (list(self.moving_objects.values()), (7, 0, 0))
        obstacles = self._detect_obstacles((7, 0, 0))
        return self._get_obs(obstacles, (7, 0, 0))

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
        agent2_state = self.state[1]
        moving_objects = list(item[1][1] for item in self.moving_objects.items())
        state = ''
        for j in range(self.max_y - 1, -1, -1):
            for i in range(self.max_x):
                if i == int(agent2_state[0]) - 1 and j == int(agent2_state[1]):
                    state += 'D'  # agent 1 - Dog
                elif i == int(agent2_state[0]) and j == int(agent2_state[1]):
                    state += 'M'  # agent 2 - Blind Man
                elif (i, j) in moving_objects:
                    state += 'O'  # moving objects
                else:
                    state += self.static_map[j][i]
            state += '\n'
        output_stream.write(state)

    def _move_object(self, moving_object_key, moving_object_value, agent_position):
        """function to change the position of the moving objects

        :param moving_object_key: key value of the moving object
        :param moving_object_value: direction of the moving object; v(vertical)/h(horizontal)
                                    and position of the object
        :param agent_position: position of the agent
        """
        if moving_object_value[0] == 'v':
            destination_position = (moving_object_value[1][0], 9) \
                if moving_object_value[1][1] == 0 \
                else (moving_object_value[1][0], moving_object_value[1][1] - 1)

        else:  # for horizontal moving objects
            destination_position = (0, moving_object_value[1][1]) \
                if moving_object_value[1][0] == self.max_x - 1 \
                else (moving_object_value[1][0] + 1, moving_object_value[1][1])

        man_position = (agent_position[0], agent_position[1])
        dog_position = (agent_position[0] - 1, agent_position[1])
        if destination_position == tuple(man_position) or \
                destination_position == tuple(dog_position) or \
                destination_position in list(value[1] for value in self.moving_objects.values()):
            pass
        else:
            self.moving_objects[moving_object_key][1] = destination_position

    def _move_agent_horizontal(self, agent_position, obstacles, action):
        """function for the horizontal movement of the agents

        :param agent_position: position of the agents
        :param obstacles: obstacles around the obstacle
        :param action: action of the agent
        :return agent_position: updated when no obstacles else the same
        """
        a_x, a_y, dx = agent_position[0], agent_position[1], -1 if action == 0 else 1 if action == 1 else 0
        nx = a_x + dx
        moving_objects = list(item[1][1] for item in self.moving_objects.items())
        dog_position = ((a_x - 1 + dx), a_y)
        if (nx < 0) or (nx >= self.max_x) or self.static_map[a_y][nx] != ' ' or (a_x - 1 + dx) < 0 or \
                (a_x - 1 + dx) >= self.max_x or self.static_map[a_y][a_x - 1 + dx] != ' ' or (
                dx == -1 and obstacles[1] == 1) or (dx == 1 and obstacles[6] == 1) \
                or (nx, a_y) in moving_objects or self._avoid_collision(dog_position, (nx, a_y)):
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
        moving_objects = list(item[1][1] for item in self.moving_objects.items())
        dog_position = (a_x - 1, ny)
        if (ny < 0) or (ny >= self.max_y) or (a_x - 1) < 0 or self.static_map[ny][a_x - 1] != ' ' or \
                self.static_map[ny][a_x] != ' ' or (
                dy == -1 and (obstacles[8] == 1 or obstacles[9] == 1)) or \
                (dy == 1 and (obstacles[3] == 1 or obstacles[4] == 1)) or \
                (a_x, ny) in moving_objects or self._avoid_collision(dog_position, (a_x, ny)):
            return agent_position
        agent_position[1] = ny
        return agent_position

    def _detect_obstacles(self, agent_position):
        """function to detect obstacles after the action of the agents is performed

        :param agent_position: position of the agents
        :return state of agent - dog (obstacles around the agents)
        """

        obstacles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        moving_objects = list(item[1][1] for item in self.moving_objects.items())
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
            elif value in moving_objects:
                obstacles[int(key)] = 1
            else:
                obstacles[int(key)] = 0
        obstacles[10] = 0
        # print('obstacles: ' + str(obstacles))
        return obstacles

    def _avoid_collision(self, dog_position, man_position):
        """function to avoid collision of the agents with moving objects

        :param dog_position: destination position of the agent - dog
        :param man_position: destination position of the agent - man
        :return: 0 if collision doesn't occur; 1 otherwise
        """
        moving_objects = list(item[1][1] for item in self.moving_objects.items())
        if dog_position in moving_objects or man_position in moving_objects:
            return 1
        else:
            return 0

    def is_goal(self):
        if (self.state[1][0], self.state[1][1]) in self.goal \
                and (self.state[1][0] - 1, self.state[1][1]) in self.goal:
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
        print('actual actions: ', actions)
        if self.noise > 0 and action1 not in (4, 5, 6):
            x = random.random()
            if x < self.noise:
                random_action = random.randrange(len(self.actions_agent2) - 1)
                action1, action2 = random_action, random_action
                print('noise actions:', action1, action2)

        agent_position = list(self.state[1])

        obstacles = self._detect_obstacles(agent_position)
        if action1 in (1, 0):
            agent_position = self._move_agent_horizontal(agent_position, obstacles, action1)
        elif action1 in (2, 3):
            agent_position = self._move_agent_vertical(agent_position, obstacles, action1)
        else:
            pass

        # moving objects action call
        for key, value in self.moving_objects.items():
            self._move_object(key, value, agent_position)

        obstacles = self._detect_obstacles(agent_position)
        if action1 in (4, 5, 6):
            obstacles[10], agent_position[2] = action1, action1
        else:
            obstacles[10], agent_position[2] = 0, 0

        self.state = (list(self.moving_objects.values()), tuple(agent_position))

        done, reward = False, 0

        if self.is_goal():
            done = True
            reward = 100
        else:
            reward = -1
        # print('returning from step: ', self.state, self._get_obs(tuple(obstacles), tuple(agent_position)), reward, done)
        return self._get_obs(tuple(obstacles), tuple(agent_position)), reward, done, {}

    # def action_values(self, actions):
    #     """
    #
    #     :param actions: list of actions
    #     :return: list of action numbers
    #     """
    #     return list(actions.index(x) for x in actions)
    #
    # def env_name(self):
    #     """
    #
    #     :return: domain name
    #     """
    #     return "GuideDog-v1"


# GUIDE_DOG = GuideDog_v1()
# GUIDE_DOG.get_moving_objects(0, 1)
# GUIDE_DOG.reset()
# GUIDE_DOG.render()
# reward = 0
# steps = 0
# print('initial state: ', GUIDE_DOG.state)
# while True:
#     action1 = int(input("Agent 1 action"))
#     action2 = int(input("Agent 2 action"))
#     obs, rew, done, info = GUIDE_DOG.step((action1, action2))
#     GUIDE_DOG.render()
#     print('after step: ', str(GUIDE_DOG.state), obs)
#     steps += 1
#     reward += rew
#     print('Reward: ', reward)
#     if done:
#         print('Goal Reached, total reward is ' + str(reward))
#         print('steps: ', steps)
#         break
