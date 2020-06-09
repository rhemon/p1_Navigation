# Project 1: Navigation

### Introduction

The following project was done as part of the Udacity's Deep Reiforcment Learning nanodegree program. The target was to train a agent to navigate through a world and pick up yellow bananas avoding blue ones. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the goal was to have the agent get an average score of +13 over 100 consecutive episodes.

### Python Version

The project uses unityagent which seems to require tensorflow and torch versions that are supported mainly by Python's 3.6.1 version, and did not seem to install with Python 3.7 up version. Therefore if you don't have python 3.6.1 installed I recommend doing so.

The library files needed for unity agent is added in the repository. You can set it up by running
```
pip install ./python
```

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

The starting code for the project was collected from there and also a lot of the code to train the DQN Agent was also gathered from previous practice projects in the nanodegree program.

### Approach

My approach to this problem was quite simple. I wanted to try out different neural network models, and play around with the parameters to run into a good performing model. Trying to train a simple agent usign experience replay, fixed q target and use epsilon greedy method to promote exploration more than exploitaiton.

Since the situation has a number of states and action, exploitation gets model stuck at a point. Some cases were ending up in a state where the agent keeps turning left and right, in another to avoid a blue banana the agent keeps going back and forth instead of avoiding it and moving away.

For experience replay, it keeps previous 100 thousand steps and takes random batch of 64 steps to trian the model and slightly updates the target model every few steps.

Compared to the code I brought from the workspaces, at the end my final changes were on the `neural network architecture`, `learning rate` and `tau`. I made the neural network much mroe deep because in most cases it seemed the smaller network wasn't fast on learning. Possibly because the deeper network is able to pick on more details compared to the smaller ones. Secondly I made the learning rate and tau much larger, to once again improve the pace of learning.

At the end with the final changes the agent achieved 13+ average score by 1200th episode. Here is a glimpse of my trained agent navigating through the world:

![The Gif's meant to be here :/](./sample.gif)

### Problems and Improvement

While the goal of the project is obtained there are multiple places to improve. To begin with there could have been significant improvement if I made use of better alogrithms like Double DQN, Duelling DQN or Prioritized Experience Replay.

Some common problem identified by me in this trained agent was there are siutations where it keeps moving forward when there is nothing ahead. It should turn around by that point to move towards a better locaiton.

The agent mostly chooses to go to its nearest yellow banana avoiding the blue ones direction in most cases, but in certain cases it could be that going through the blue banana would end up resulting it to get even more yellow bananas. While discount rate during training I used was already high, but it could posibly been improved if we did Prioritized Experience Replay.

For reducing the moves toward a empty area, we could introduce a small penalty for moves where you dont collect anything. This could help the agent to learn to pick more bananas in fewer steps.

### Thank you

It was only possible for me to do the project because of all the resources provided by Udacity. Starting from the amazing environment to the starting code for training the agent greatly helped me to quickly get into actually playing with the agent. I hope to explore more once I am done with nanodegree program.

