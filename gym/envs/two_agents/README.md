# Environments with two-agents

To call the environment, use - `gym.make("GuidedNavigation-v0/GuidedNavigation-v1")`

<<<<<<< HEAD
=======
## Environment 1: GuidedNavigation-v0

The environment is an 8x10 grid which has two agents in which one agent guides the other agent in the environment to reach the home base.

![Alt text](envs/two_agents/images/GuidedNavigation-v0.jpg?raw=true "GuidedNavigation - v0")<br />
Actions for each agent: <br />
```['LEFT', 'RIGHT', 'UP', 'DOWN']```<br />
Action Space: <br />
```Tuple([Discrete(Action set), Discrete(Action set)])```<br />
#####Observation Space<br />
######Agent 1<br />
```Agent 1 can observe the surroundings around both the agents marked by the positions 0 - 9 in the figure below and detenct any obstacles in these positions```<br/>

|     :---:     |     :---:     |     :---:     |     :---:     |
|       2       |       3       |       4       |       5       |
|       1       |     Agent1    |     Agent2    |       6       |
|       0       |       9       |       8       |       7       |

Observation<br/>
Type: Box(10)
    Num	     Observation                 Min         Max
    0-9	     Position 0-9 Obstacles       0           1


######Agent 2<br />
```Agent 2 can always identify its position ```<br/>
Observation<br/>
Type: Box(10)
    Num	     Observation       Min         Max
     0	     x-coordinate       0           8
     1	     y-coordinate       0           10


## Environment 2: GuidedNavigation-v1

![Alt text](envs/two_agents/images/GuidedNavigation-v1.jpg?raw=true "GuidedNavigation - v1")
>>>>>>> Adding images and README update
