Environments with two-agents
****************************
.. contents:: **Environments with two-agents**
   :depth: 1


.. contents:: **Usage:**
   :depth: 2

.. code:: python

    import gym
    env = gym.make('GuidedNavigation-v0/GuidedNavigation-v1')
    env.reset()
    env.render()

.. contents:: **Environment 1: GuidedNavigation-v0**
   :depth: 2

GuidedNavigation is an 8x10 grid which has two agents in which one agent guides the other agent in the environment to reach the home base.

![Alt text](envs/two_agents/images/GuidedNavigation-v0.jpg?raw=true "GuidedNavigation - v0")

.. contents:: ***Actions for each agent:***
   :depth: 3
```['LEFT', 'RIGHT', 'UP', 'DOWN']```

.. contents:: ***Action Space:***
   :depth: 3
```Tuple([Discrete(Action set), Discrete(Action set)])```

.. contents:: ***Observation Space:***
   :depth: 3

.. contents:: ***Agent 1:***
   :depth: 4

Agent 1 can observe the surroundings around both the agents marked by the positions 0 - 9 in the figure below and detenct any obstacles in these positions

+---------+---------+-----------+-----------+
| 2       |  3      |  4        |  5        |
+---------+---------+-----------+-----------+
| 1       |  Agent1 |   Agent2  |  6        |
+---------+---------+-----------+-----------+
| 0       |  9      |  8        |  7        |
+---------+---------+-----------+-----------+


Observation
Type: Box(10)
+---------+-------------------------+-----------+-----------+
| Num     | Observation             |  Min      |  Max      |
+---------+-------------------------+-----------+-----------+
| 0-9     | Position 0-9 Obstacles  |  0        |  1        |
+---------+-------------------------+-----------+-----------+

.. contents:: ***Agent 2:***
   :depth: 4

Agent 2 can always identify its position

Observation

Type: Box(10)

+---------+-----------------+-----------+-----------+
| Num     | Observation     |  Min      |  Max      |
+---------+-----------------+-----------+-----------+
| 0       | x-coordinate    |  0        |  8        |
+---------+-----------------+-----------+-----------+
| 1       | y-coordinate    |  0        |  10       |
+---------+-----------------+-----------+-----------+


.. contents:: **Environment 2: GuidedNavigation-v1**
   :depth: 2

![Alt text](envs/two_agents/images/GuidedNavigation-v1.jpg?raw=true "GuidedNavigation - v1")
