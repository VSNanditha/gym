# 2-agent block dudes environment.
# =================================
import random
import sys

import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple, Box
# from gym import spaces
from six import StringIO


class BlockDudes(Env):
    def __init__(self):
        self.noise = 0.05  # (executed actions) != (intended actions) with this probability
        self.MaxY = 6
        self.WorldYVals = [3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3]
        self.MaxX = len(self.WorldYVals)
        self.StaticMap = {}
        for i in range(self.MaxX):
            if i not in self.StaticMap:
                self.StaticMap[i] = {}
            for j in range(self.MaxY - 1, -1, -1):
                if j >= self.WorldYVals[i]:
                    self.StaticMap[i][j] = 0
                else:
                    self.StaticMap[i][j] = 1

        self.Goal1, self.Goal2 = (0, 3), (1, 3)
        self.actions = ['up', 'west', 'east', 'pickup', 'putdown']
        self.action_space = Tuple(
            [Discrete(len(self.actions)), Discrete(len(self.actions))]
        )
        o12 = self.reset()
        self.low = np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -50, 0, -1])
        self.high = np.array(
            [self.MaxX, self.MaxY, 1, 2, self.MaxX, self.MaxY, self.MaxX, self.MaxY, self.MaxX, self.MaxY, 50, 2, 50])
        self.observation_space = Tuple(
            [Box(self.low, self.high, dtype=np.int), Box(self.low, self.high, dtype=np.int)]
        )

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        self._print_state(outfile)

    def _print_state(self, f):
        s1, s2 = self.state[0], self.state[1]
        b_array = s1[4]
        for j in range(self.MaxY - 1, -1, -1):
            s = ''
            for i in range(self.MaxX):
                bind = self._getBlockAt(b_array, i, j)
                if (bind >= 0):
                    s += str(bind)  # Movable block, with ID=bind
                elif i == s1[0] and j == s1[1] and self.StaticMap[i][j] == 0:
                    if s1[2]:
                        s += '<'  # agent 1 facing left
                    else:
                        s += '>'  # agent 1 facing right
                elif i == s2[0] and j == s2[1] and self.StaticMap[i][j] == 0:
                    if s2[2]:
                        s += '{'  # agent 2 facing left
                    else:
                        s += '}'  # agent 2 facing right
                elif self.StaticMap[i][j] == 0:
                    s += ' '  # empty space
                else:
                    s += 'B'  # Immovable block
            s += '\n'
            f.write(s)

    def _getBlockAt(self, b_array, x, y):  # return block_id at (x,y), -1 if none
        ret_val = -1
        for i in range(len(b_array)):
            bx, by = b_array[i][0], b_array[i][1]
            if (bx == x) and (by == y):
                ret_val = i
        return ret_val

    def _putBlockAt(self, b_array, x, y, bind):  # return modified block_array
        b_list = list(b_array)
        b_list[bind] = (x, y)
        return tuple(b_list)

    def _pickupBlock(self, s, other_x,
                     other_y):  # return (possibly) modified state --(succ (T/F), changed_b_array, which_block_pickdup (-1 if fail))
        ret_s = list(s)
        if (s[3] >= 0):  # already holding a block
            return tuple(ret_s)  # no change
        if (s[2]):
            x = s[0] - 1
        else:
            x = s[0] + 1
        bind = self._getBlockAt(s[4], x, s[1])
        if bind < 0:  # no block at (x, s[1])
            return tuple(ret_s)  # no change
        elif other_x == x:  # other agent in block's column; unsafe op -> disallow
            return tuple(ret_s)  # no change
        else:
            if self._getBlockAt(s[4], x, s[1] + 1) >= 0:  # block above block
                return tuple(ret_s)  # no change
            elif self.StaticMap[x][s[1] + 1] == 1:  # brick above block
                return tuple(ret_s)  # no change
            else:
                ret_s[3] = bind
                ret_s[4] = self._putBlockAt(s[4], s[0], s[1] + 1, bind)
                return tuple(ret_s)

    def _putdownBlock(self, s, other_x, other_y):  # return (possibly) modified state --(succ (T/F), changed_b_array)
        ret_s = list(s)
        if (s[3] < 0):  # not holding a block
            return tuple(ret_s)  # no change
        if (s[2]):
            nx = s[0] - 1
        else:
            nx = s[0] + 1
        if (nx < 0) or (nx >= self.MaxX):  # beyond world boundary
            return tuple(ret_s)  # no change
        if (other_x == nx):  # other agent in nx column; unsafe op -> disallow
            return tuple(ret_s)  # no change
        ht = self._greatestHeightBelow(nx, s[1] + 1, s[4])
        if (ht > s[1]):  # cannot drop block if walled off from throw position
            return tuple(ret_s)  # no change

        bind = self._getBlockAt(s[4], s[0], s[1] + 1)
        if bind != s[3]:  # Something wrong
            # print "WHAT?",bind,"<---->",s
            return tuple(ret_s)  # no change
        else:
            ret_s[3] = -1
            ret_s[4] = self._putBlockAt(s[4], nx, ht + 1, bind)
            return tuple(ret_s)

    def _moveUp(self, s, other_x, other_y):  # return (possibly) modified state
        ret_s = list(s)
        ax, ay, dx = s[0], s[1], -1 if s[2] else 1
        nx, ny = ax + dx, ay + 1
        if (nx < 0) or (nx >= self.MaxX):
            return tuple(ret_s)
        if (nx == other_x):  # other agent in destination column; agents must not cross -> disallow
            return tuple(ret_s)
        clearing = ny + 1 if s[3] >= 0 else ny
        htAtNX = self._greatestHeightBelow(nx, clearing, s[4])
        if htAtNX != ay:
            return tuple(ret_s)
        ret_s[0], ret_s[1] = nx, ny
        if (s[3] >= 0):
            ret_s[4] = self._putBlockAt(s[4], nx, ny + 1, s[3])
        return tuple(ret_s)

    def _moveHorizontally(self, s, a_ind, other_x, other_y):  # return (possibly) modified state
        ret_s = list(s)
        if ((s[2] and (a_ind == 2)) or ((not s[2]) and (a_ind == 1))):  # only toggle heading
            ret_s[2] = not s[2]
            return tuple(ret_s)
        ax, ay, dx = s[0], s[1], -1 if s[2] else 1
        nx = ax + dx
        if (nx < 0) or (nx >= self.MaxX):
            return tuple(ret_s)
        if (nx == other_x):  # other agent in destination column; agents must not cross -> disallow
            return tuple(ret_s)
        htAtNX = self._greatestHeightBelow(nx, ay, s[4])
        if htAtNX >= ay:
            return tuple(ret_s)
        ny = htAtNX + 1
        ret_s[0], ret_s[1] = nx, ny
        if (s[3] >= 0):
            ret_s[4] = self._putBlockAt(s[4], nx, ny + 1, s[3])
        return tuple(ret_s)

    def _greatestHeightBelow(self, nx, mxY, b_array):
        mxHt = -1  # BB-correct?
        for j in range(self.MaxY - 1, -1, -1):
            if self.StaticMap[nx][j] == 1:
                mxHt = j
                break
        if (mxHt < mxY):
            for (bx, by) in b_array:
                if (bx == nx):
                    if (by > mxHt) and (by <= mxY):
                        mxHt = by
        return mxHt

    def reset(self):
        b_array = ((16, 3), (13, 3), (4, 0))
        x = random.random()
        if x < 0.5:
            self.state = ((4, 1, True, -1, b_array), (14, 3, True, -1, b_array))
        else:
            self.state = ((14, 3, True, -1, b_array), (4, 1, True, -1, b_array))

        return self._get_obs()

    def _getPrivateObservations(self, s,
                                s_other):  # see (x,y, heading, holding, nearest block ID & x-distance, other_dist, other_carrying if <=4)
        other_dist = (s[0] - s_other[0])
        if other_dist > 4:
            other_dist = 50
        elif other_dist < -4:
            other_dist = -50
        other_carrying = s_other[3] if abs(other_dist) <= 4 else 50
        other_heading = 1 if s_other[2] else 0

        # min_dist = 50
        ret_list = [s[0], s[1], 1 if s[2] else 0, s[3]]
        for bid in range(len(s[4])):
            (bx, by) = s[4][bid]
            ret_list.append(bx)
            ret_list.append(by)
            # dist = abs(bx-s[0])
        ret_list += [other_dist, other_heading if abs(other_dist) <= 4 else 2, other_carrying]
        return tuple(ret_list)

    def _get_obs(self):
        return (np.array(self._getPrivateObservations(self.state[0], self.state[1])),
                np.array(self._getPrivateObservations(self.state[1], self.state[0])))

    def _isGoal(self):
        s1, s2 = self.state[0], self.state[1]
        if ((s1[0], s1[1]) == self.Goal1) and ((s2[0], s2[1]) == self.Goal2) or ((s1[0], s1[1]) == self.Goal2) and (
                (s2[0], s2[1]) == self.Goal1):
            return True
        else:
            return False

    def step(self, actions):
        assert self.action_space.contains(actions), "%r (%s) invalid" % (actions, type(actions))
        a1, a2 = actions
        s1, s2 = self.state[0], self.state[1]

        temp_s2 = list(s2)
        a1_ind, a2_ind = a1, a2
        if self.noise > 0:
            x = random.random()
            if (x < self.noise):
                a1_ind, a2_ind = random.randrange(len(self.actions) - 1), random.randrange(len(self.actions) - 1)

        if a1_ind == 0:
            ns1 = self._moveUp(s1, s2[0], s2[1])
        if (a1_ind == 1) or (a1_ind == 2):
            ns1 = self._moveHorizontally(s1, a1_ind, s2[0], s2[1])
        if a1_ind == 3:
            ns1 = self._pickupBlock(s1, s2[0], s2[1])
        if a1_ind == 4:
            ns1 = self._putdownBlock(s1, s2[0], s2[1])

        if (ns1[0] == s2[0]) and (ns1[1] == s2[1]):  # Prohibits (even temporary) collision
            # return (s1,s2)
            ns1 = s1

        temp_ns1 = list(ns1)
        temp_s2[4] = ns1[4]
        if a2_ind == 0:
            ns2 = self._moveUp(tuple(temp_s2), ns1[0], ns1[1])
        if (a2_ind == 1) or (a2_ind == 2):
            ns2 = self._moveHorizontally(tuple(temp_s2), a2_ind, ns1[0], ns1[1])
        if a2_ind == 3:
            ns2 = self._pickupBlock(tuple(temp_s2), ns1[0], ns1[1])
        if a2_ind == 4:
            ns2 = self._putdownBlock(tuple(temp_s2), ns1[0], ns1[1])

        temp_ns1[4] = ns2[4]

        if (ns1[0] == ns2[0]) and (ns1[1] == ns2[1]):  # Prohibits (even temporary) collision
            # return (s1,s2)
            ns2 = temp_s2  # physical collision means block map couldn't have changed between A1's action and now.

        self.state = (tuple(temp_ns1), ns2)
        done, reward = False, -1
        if self._isGoal():
            done = True
            reward = 100
        return self._get_obs(), reward, done, {}

    def action_values(self, actions):
        return list(actions.index(x) for x in actions)

    def env_name(self):
        return "BlockDudes-v0"
