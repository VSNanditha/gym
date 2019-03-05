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

GuidedNavigation is an 8x10 grid which has two agents in which one agent guides the other agent in the environment to reach the home base. This version of the domain has only static obstacles.

.. image:: https://github.com/VSNanditha/gym/blob/master/gym/envs/two_agents/images/GuidedNavigation-v0.png
   :height: 50px
   :width: 50 px
   :scale: 10 %
   :alt:  GuidedNavigation-v0
   :align: right

Action Space:
-------------
``Tuple([Discrete(4), Discrete(4)])``

Observation Space:
------------------
``Tuple([Box(10), Box(2)])``

Agent 1:
-----------------------

Actions:
^^^^^^^^

``['LEFT', 'RIGHT', 'UP', 'DOWN']``

Observations:
^^^^^^^^^^^^

Type: Box(10)

+---------+-------------------------+-----------+-----------+
| Num     | Observation             |  Min      |  Max      |
+---------+-------------------------+-----------+-----------+
| 0-9     | Position 0-9 Obstacles  |  0        |  1        |
+---------+-------------------------+-----------+-----------+

Agent 2:
-----------------------

Actions:
^^^^^^^^

``['LEFT', 'RIGHT', 'UP', 'DOWN']``

Observations:
^^^^^^^^^^^^

Type: Box(2)

+---------+-----------------+-----------+-----------+
| Num     | Observation     |  Min      |  Max      |
+---------+-----------------+-----------+-----------+
| 0       | x-coordinate    |  0        |  8        |
+---------+-----------------+-----------+-----------+
| 1       | y-coordinate    |  0        |  10       |
+---------+-----------------+-----------+-----------+

Environment 2: GuidedNavigation-v1
==================================

GuidedNavigation is an 8x10 grid which has two agents in which one agent guides the other agent in the environment to reach the home base. This version of the domain has static obstacles and also a dynamic obstacle which moves in a fixed path vertically.

.. image:: https://github.com/VSNanditha/gym/blob/master/gym/envs/two_agents/images/GuidedNavigation-v1.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt:  GuidedNavigation-v1
   :align: right

Action Space:
-------------
``Tuple([Discrete(7), Discrete(4)])``

Observation Space:
------------------
``Tuple([Box(11), Box(3)])``

Agent 1:
-----------------------

Actions:
^^^^^^^^

``['LEFT', 'RIGHT', 'UP', 'DOWN', 'SEND-TUG', 'SEND-PUSH', 'SEND-DOWN']``

Observations:
^^^^^^^^^^^^

Type: Box(11)

+---------+-------------------------+-----------+-----------+
| Num     | Observation             |  Min      |  Max      |
+---------+-------------------------+-----------+-----------+
| 0-9     | Position 0-9 Obstacles  |  0        |  1        |
+---------+-------------------------+-----------+-----------+
| 10      | Communication           |  0        |  3        |
+---------+-------------------------+-----------+-----------+

Communication
"""""""""""""

0 - No communication

1 - Send Tug

2 - Send Push

3 - Send Down

Agent 2:
-----------------------

Actions:
^^^^^^^^

``['LEFT', 'RIGHT', 'UP', 'DOWN']``

Observations:
^^^^^^^^^^^^

Type: Box(3)

+---------+-----------------+-----------+-----------+
| Num     | Observation     |  Min      |  Max      |
+---------+-----------------+-----------+-----------+
| 0       | x-coordinate    |  0        |  8        |
+---------+-----------------+-----------+-----------+
| 1       | y-coordinate    |  0        |  10       |
+---------+-----------------+-----------+-----------+
| 2       | Communication   |  0        |  3        |
+---------+-----------------+-----------+-----------+

Communication
"""""""""""""

0 - No communication

1 - Receive Tug

2 - Receive Push

3 - Receive Down
