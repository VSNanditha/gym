# 2-agent object transport environment.
# =================================

import random
import sys

import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple, Box
from six import StringIO


class ObjectTransport(Env):
    def __init__(self):
        self.noise = 0.1  # (executed actions) != (intended actions) with this probability
        self.MaxY = 9
        self.ImmovableBlocks = {
            '8': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            '5': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            '1': [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            '0': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        self.MaxX = 16
        self.StaticMap = {}
        for Y in range(self.MaxY):
            if Y not in self.StaticMap:
                self.StaticMap[Y] = [0] * self.MaxX
            if str(Y) in self.ImmovableBlocks.keys():
                for X in range(self.MaxX):
                    if self.ImmovableBlocks[str(Y)][X] == 1:
                        self.StaticMap[Y][X] = 1
                    else:
                        self.StaticMap[Y][X] = 0

        """
        for i in range(self.MaxY-1, -1, -1):
            for j in range(self.MaxX):
                print(i, j)
                print(str(self.StaticMap[i][j]), end=' ')
            print("\n")
        """

        self.Goal = [(5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8)]
        self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'STAND-STILL']
        self.action_space = Tuple(
            [Discrete(len(self.actions)), Discrete(len(self.actions))]
        )
        self.reset()
        self.low = np.array([0, 0, 0])
        self.high = np.array([self.MaxX, self.MaxY, 2])
        self.observation_space = Tuple(
            [Box(self.low, self.high, dtype=np.int), Box(self.low, self.high, dtype=np.int)]
        )
        self.object = ()

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        self._print_state(outfile)

    def _print_state(self, f):
        s1, s2 = self.state[0], self.state[1]
        b_array = self.object
        print('b array: ', b_array)
        for j in range(self.MaxY - 1, -1, -1):
            s = ''
            for i in range(self.MaxX):
                bind = self._getBlockAt(b_array, i, j)
                if bind >= 0:
                    s += 'M'  # Movable block
                elif i == int(s1[0]) and j == int(s1[1]) and self.StaticMap[j][i] == 0:
                    s += 'A'  # agent 1
                elif i == int(s2[0]) and j == int(s2[1]) and self.StaticMap[j][i] == 0:
                    s += 'B'  # agent 2
                elif self.StaticMap[j][i] == 0:
                    s += ' '  # empty space
                else:
                    s += 'X'  # Immovable block
            s += '\n'
            f.write(s)

    def _getBlockAt(self, b_array, x, y):  # return block_id at (x,y), -1 if none
        ret_val = -1
        bx, by = b_array[0], b_array[1]
        if (bx == x) and (by == y):
            ret_val = 1
        return ret_val

    def reset(self):
        self.object = (8, 4)  # position of movable object
        self.state = ((1, 0, 0), (15, 0, 0))
        return self._get_obs()

    def _get_obs(self):
        s1, s2 = list(self.state[0]), list(self.state[1])
        return np.array(tuple(s1)), np.array(tuple(s2))

    def _moveHorizontally(self, s, a_ind):  # return (possibly) modified state
        ret_s = list(s)
        ax, ay, dx = s[0], s[1], -1 if a_ind == 0 else 1 if a_ind == 1 else 0
        nx = ax + dx
        if (nx < 0) or (nx >= self.MaxX) or self.StaticMap[ay][nx] != 0 or (nx, ay) == self.object:
            return tuple(ret_s)
        ret_s[0] = nx
        if ret_s[0] == self.object[0] + 1 and ret_s[1] == self.object[1]:
            ret_s[2] = 2
        elif ret_s[0] == self.object[0] - 1 and ret_s[1] == self.object[1]:
            ret_s[2] = 1
        else:
            ret_s[2] = 0
        return tuple(ret_s)

    def _moveVertically(self, s, a_ind):  # return (possibly) modified state
        ret_s = list(s)
        ax, ay, dy = s[0], s[1], -1 if a_ind == 3 else 1 if a_ind == 2 else 0
        ny = ay + dy
        if (ny < 0) or (ny >= self.MaxY) or self.StaticMap[ny][ax] != 0 or (ax, ny) == self.object:
            return s
        ret_s[1] = ny
        if ret_s[0] == self.object[0] + 1 and ret_s[1] == self.object[1]:
            ret_s[2] = 2
        elif ret_s[0] == self.object[0] - 1 and ret_s[1] == self.object[1]:
            ret_s[2] = 1
        else:
            ret_s[2] = 0
        return tuple(ret_s)

    def _isGoal(self):
        if tuple(self.object) in self.Goal:
            return True
        else:
            return False

    def _moveBlock(self, s1, s2, a_ind):
        ret_s1, ret_s2 = list(s1), list(s2)
        ret_block = list(self.object)
        dx = -1 if a_ind == 0 else 1 if a_ind == 1 else 0
        dy = -1 if a_ind == 3 else 1 if a_ind == 2 else 0
        if dx != 0:
            ret_s1[0] += dx
            ret_s2[0] += dx
            ret_block[0] += dx
        else:
            ret_s1[1] += dy
            ret_s2[1] += dy
            ret_block[1] += dy

        if (ret_s1[0] < 0 or ret_s1[0] >= self.MaxX or ret_s2[0] < 0 or ret_s2[0] >= self.MaxX or ret_s1[1] < 0 or
            ret_s1[1] >= self.MaxY or ret_s2[1] < 0 or ret_s1[1] >= self.MaxY) or (not (
                ret_s1[0] < 0 or ret_s1[0] >= self.MaxX or ret_s2[0] < 0 or ret_s2[0] >= self.MaxX or ret_s1[1] < 0 or
                ret_s1[1] >= self.MaxY or ret_s2[1] < 0 or ret_s1[1] >= self.MaxY) and (self.StaticMap[ret_s1[1]][
                                                                                            ret_s1[0]] != 0 or
                                                                                        self.StaticMap[ret_s2[1]][
                                                                                            ret_s2[0]])):
            return s1, s2, self.object
        return tuple(ret_s1), tuple(ret_s2), ret_block

    def step(self, actions):
        # assert self.action_space.contains(actions), "%r (%s) invalid" % (actions, type(actions))
        a1, a2 = actions
        s1, s2 = self.state[0], self.state[1]
        block = self.object

        if self.noise > 0:
            x = random.random()
            if x < self.noise:
                a1, a2 = random.randrange(len(self.actions) - 1), random.randrange(len(self.actions) - 1)

        if s1[2] != 0 and s2[2] != 0 and a1 == a2:
            ns1, ns2, self.object = self._moveBlock(s1, s2, a1)
        elif s1[2] != 0 and s2[2] != 0 and a1 != a2:
            ns1, ns2 = s1, s2
        else:
            if s1[2] == 0:
                if (a1 == 0) or (a1 == 1):
                    ns1 = self._moveHorizontally(s1, a1)
                elif a1 == 2 or a1 == 3:
                    ns1 = self._moveVertically(s1, a1)
                else:
                    ns1 = s1
            else:
                ns1 = s1

            if s2[2] == 0:
                if (a2 == 0) or (a2 == 1):
                    ns2 = self._moveHorizontally(s2, a2)
                elif a2 == 2 or a2 == 3:
                    ns2 = self._moveVertically(s2, a2)
                else:
                    ns2 = s2
            else:
                ns2 = s2

        done, reward = False, [0, 0]
        if (ns1[0] == ns2[0]) and (ns1[1] == ns2[1]):  # Prohibits collision
            return self._get_obs(), tuple(reward), done, {}

        self.state = tuple(ns1), tuple(ns2)

        if self._isGoal():
            done = True
            reward = [100, 100]
        else:
            if ((ns1[0] == self.object[0] + 1 or ns1[0] == self.object[0] - 1) and ns1[1] == self.object[1]) and (
                    (s1[0] != block[0] + 1 and s1[0] != block[0] - 1) or s1[1] != block[1]):
                reward[0] = 10
            else:
                reward[0] = -1

            if ((ns2[0] == self.object[0] + 1 or ns2[0] == self.object[0] - 1) and ns2[1] == self.object[1]) and (
                    (s2[0] != block[0] + 1 and s2[0] != block[0] - 1) or s2[1] != block[1]):
                reward[1] = 10
            else:
                reward[1] = -1
        return self._get_obs(), tuple(reward), done, {}

    def action_values(self, actions):
        return list(actions.index(x) for x in actions)

    def env_name(self):
        return "ObjectTransport-v0"

# objtransport = ObjectTransport()
# objtransport.reset()
# objtransport.render()
# reward0 = 0
# reward1 = 0
# total_reward = 0
# steps = 0
# while True:
#     a1 = int(input("Agent 1 action"))
#     a2 = int(input("Agent 2 action"))
#     obs, rew, done, info = objtransport.step((a1, a2))
#     print(objtransport.state)
#     objtransport.render()
#     steps += 1
#     rew = list(rew)
#     reward0 += rew[0]
#     reward1 += rew[1]
#     total_reward += (rew[0] + rew[1])
#     print('Reward: ', reward0, reward1)
#     print('total: ', total_reward)
#     if objtransport._isGoal():
#         print('Goal Reached, total reward is ' + str(total_reward))
#         print('steps: ', steps)
#         break;
