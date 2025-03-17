Link to the video:
https://youtu.be/gjaC4GfFOQI

Link to the article:
https://medium.com/@curiositycreates91/dqn-algorithm-implementation-for-inverted-pendulum-from-simulation-to-physical-system-d57c01d0fb90

Dependencies:
    Gymnasium
    pytorch
    PyBullet

This repository contains the code that applies Deep Q-learning (DQN) reinforcing learng algorithm to a classic control problem, balancing an inverted pendulum. The algorithm is applied to the problem in 3 progressively difficult environments: 1- Gymnasium "CartPole-v1", 2- A real-time environment modeled in Pyullet, and 3- A real life inverted pendulum.

The aim is to show how an agent trained mostly in simulation performs in a real world system. 

gym_inverted_pendulum_mc.py and gym_inverted_pendulum_mc.py contains the code to train an agent to balance the inverted pendulum (or CartPole-v1) in Gymnasium.
If you have PyBullet and pytorch installed, you should be able to clone the repo and directly run the "demo_swingup.py" file to watch the demo of the inverted pendulum being balanced in PyBullet. For real life system, of course, hardware is needed.

saved_models\run2_swingup contains the saved model and videos of the agent at different stages of training.

TBD:
    1. Wire diagram for the real life system
    2. SolidWorks files of the system