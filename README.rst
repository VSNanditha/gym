Environments with two-agents
****************************

Usage:
======

.. code:: python

    import gym
    env = gym.make('GuidedNavigation-v0/GuidedNavigation-v1')
    env.reset()
    env.render()

Environment 1: GuidedNavigation-v0
==================================

GuidedNavigation is an 8x10 grid which has two agents in which one agent guides the other agent in the environment to reach the home base.

.. image:: ./images/GuidedNavigation-v0.jpg

Actions for each agent:
-----------------------
```['LEFT', 'RIGHT', 'UP', 'DOWN']```

Action Space:
-------------
```Tuple([Discrete(Action set), Discrete(Action set)])```

Observation Space:
------------------

Agent 1:
^^^^^^^^

Agent 1 can observe the surroundings around both the agents marked by the positions 0 - 9 in the figure below and detenct any obstacles in these positions

+---------+---------+-----------+-----------+
|    2    |    3    |     4     |     5     |
+---------+---------+-----------+-----------+
|    1    |  Agent1 |   Agent2  |     6     |
+---------+---------+-----------+-----------+
|    0    |    9    |     8     |     7     |
+---------+---------+-----------+-----------+


Observation
"""""""""""

Type: Box(10)
+---------+-------------------------+-----------+-----------+
| Num     | Observation             |  Min      |  Max      |
+---------+-------------------------+-----------+-----------+
| 0-9     | Position 0-9 Obstacles  |  0        |  1        |
+---------+-------------------------+-----------+-----------+

Agent 2:
^^^^^^^^

Agent 2 can always identify its position

Observation
"""""""""""

Type: Box(10)

+---------+-----------------+-----------+-----------+
| Num     | Observation     |  Min      |  Max      |
+---------+-----------------+-----------+-----------+
| 0       | x-coordinate    |  0        |  8        |
+---------+-----------------+-----------+-----------+
| 1       | y-coordinate    |  0        |  10       |
+---------+-----------------+-----------+-----------+


Environment 2: GuidedNavigation-v1
==================================

.. image:: ./images/GuidedNavigation-v1.jpg
