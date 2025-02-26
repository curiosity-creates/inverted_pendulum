import pybullet as p
import torch
import torch.optim as optim
import numpy as np
import pybullet_data
import time
from itertools import count
import math
import os

import matplotlib.pyplot as plt

import torch.optim.rmsprop
import torch.optim.rmsprop

from gym_inverted_pendulum import FCQ
from gym_inverted_pendulum import ReplayBuffer

class pybulletDQN():
    def __init__(self, replay_buffer, online_model, target_model, optimizer, update_freq, epochs = 40, gamma = 1):
        self.replay_buffer = replay_buffer
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optimizer
        # self.warmup_batches = warmup_batches
        self.update_freq = update_freq
        self.gamma = gamma
        self.epoch = epochs

    def choose_action_egreedy(self, state, eps):
        state = torch.tensor(state, dtype=torch.float32, device=self.online_model.device)
        
        with torch.no_grad():
            q = self.online_model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > eps:
            action = np.argmax(q)
        else:
            action = np.random.randint(self.online_model.output_dim)

        return action
    
    def choose_action_greedy(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.online_model.device)
        
        with torch.no_grad():
            q = self.online_model(state).cpu().detach()

        q = q.numpy()

        action = np.argmax(q)

        return action
    

    def soft_update_weights(self, tau=1):
        # """
        # Performs a soft update of the target model weights.

        # Args:
        #     online_model: The source model to copy weights from.
        #     target_model: The destination model to copy weights to.
        #     tau: The interpolation factor (a value between 0 and 1).
        #         tau = 1 corresponds to a hard update (direct copy).
        #         tau << 1 corresponds to a slow, gradual update.
        # """

        for target_param, online_param in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        states, actions, rewards, next_states, terminals = self.replay_buffer.draw_samples()

        states = torch.tensor(states, dtype=torch.float32, device=self.online_model.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.online_model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.online_model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.online_model.device)
        terminals = torch.tensor(terminals, dtype=torch.float32, device=self.online_model.device)

        qsa_next_max = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        yj = rewards + (self.gamma * qsa_next_max * (1 - terminals))

        qsa = self.online_model(states).gather(1,actions)

        td_error = yj - qsa
        loss = td_error.pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def eval(self, robotId, max_steps = 500, randomize_val = 0.17, friction = 0.002):
        reset_joint_swingup(robotId, randomize_val= randomize_val, friction=friction)
        state = get_state(robotId)
        episode_score = 0

        terminal = False
        truncated = False

        for step in count():
            state = get_state(robotId)
            # print(f"State: {state}")
            # if state[2] > 0.7 and state[2] < 5.58:
            #     reward = 0
            #     terminal = True
            #     # print("Terminal due to joint 2")
            if state[0] > 3.14 or state[0] < -3.14:
                reward = -100
                terminal = True
                # print("Terminal due to joint 1")
            else:
                reward = math.cos(state[2]) - 0.002 * (state[3]) ** 2 + 0.001 * math.cos(state[0])

            if step > max_steps:
                truncated = True
                # print("Truncated")
            
            if step > 0:
                episode_score += reward

            if terminal or truncated:
                action = -1
                send_action(robotId, action)
                print(f"Eval episode score: {episode_score}")
                break

            action = self.choose_action_greedy(state)
            send_action(robotId, action)

            time.sleep(0.02)

        return episode_score

def reset_joint(robotId, randomize_val= 0.17, friction = 0.002):
    randomize1 = np.random.uniform(-randomize_val, randomize_val)              # Randomize the initial joint position. Angle is in radians
    randomize2 = np.random.uniform(-randomize_val, randomize_val)              # Randomize the initial joint position. Angle is in radians
    p.resetJointState(bodyUniqueId=robotId, jointIndex=1, targetValue=randomize1)
    p.resetJointState(bodyUniqueId=robotId, jointIndex=2, targetValue=3.14 + randomize2)
    p.setJointMotorControl2(bodyUniqueId=robotId, 
                                jointIndex=2, 
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity = 0,
                                force = friction)
    

def reset_joint_swingup(robotId, randomize_val= 0.17, friction = 0.002):
    randomize1 = np.random.uniform(-randomize_val, randomize_val)              # Randomize the initial joint position. Angle is in radians
    randomize2 = np.random.uniform(-randomize_val, randomize_val)              # Randomize the initial joint position. Angle is in radians
    p.resetJointState(bodyUniqueId=robotId, jointIndex=1, targetValue=randomize1)
    p.resetJointState(bodyUniqueId=robotId, jointIndex=2, targetValue=randomize2)
    p.setJointMotorControl2(bodyUniqueId=robotId, 
                                jointIndex=2, 
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity = 0,
                                force = friction)
    
    
def get_state(robotId):
    deg = 2 * math.pi
    # print(p.getJointState(robotId, 2))
    joint2_state = math.fmod(math.fmod(p.getJointState(robotId, 2)[0]-3.14, deg) + deg, deg)
    return np.array((p.getJointState(robotId, 1)[0], p.getJointState(robotId, 1)[1], joint2_state, p.getJointState(robotId, 2)[1]))

def send_action(robotId, action):
    if action == 0:
        p.setJointMotorControl2(bodyUniqueId=robotId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity = -5,
                                force = 0.3)
    elif action == 1:
        p.setJointMotorControl2(bodyUniqueId=robotId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity = 5,
                                force = 0.3)
    elif action == -1:
        # print("Action 2")
        p.setJointMotorControl2(bodyUniqueId=robotId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity = 0,
                                force = 0.3)



if __name__ == "__main__":

    epochs = 40
    episodes = 8000
    max_steps = 500

    learning_rate = 0.0005
    batch_size = 256
    warmup_batches = 5
    update_freq = 10                                            # Number signifies steps after which target network will be updated
    min_steps_to_learn = warmup_batches * batch_size

    replay_buffer = ReplayBuffer(batch_size=batch_size)

    num_actions = 2
    num_states = 4
    model_path = os.path.join("saved_models/run24", "model_5000.pth")
    if os.path.exists(model_path):
        print("Loading existing model")
        online_model = torch.load(model_path)
        target_model = torch.load(model_path)
    else:
        print("Creating new model")
        online_model = FCQ(num_states, num_actions, (256, 128))
        target_model = FCQ(num_states, num_actions, (256, 128))
    optimizer = optim.RMSprop(online_model.parameters(), lr=learning_rate)

    agent = pybulletDQN(replay_buffer, online_model, target_model, optimizer, update_freq, epochs=epochs, gamma=0.99)
    agent.soft_update_weights()

    eval = True
    eval_interval = 50   

    decay_steps = episodes * 0.8
    eval_scores = []
    episode_scores = []

    save_intervals = 1000
    save_dir = "saved_models/run24"

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    friction = 0.0047
    randomize_val = 0.05
    planeId = p.loadURDF("plane.urdf", [0, 0, 0])
    robotId_path = os.path.join("urdf and meshes/with smaller weight/rlr_urdf.urdf")
    robotId = p.loadURDF(robotId_path, [0, 0, 0])

    # p.resetJointState(bodyUniqueId=robotId, jointIndex=2, targetValue=3.14)
    # p.setJointMotorControl2(bodyUniqueId=robotId, 
    #                             jointIndex=2, 
    #                             controlMode=p.VELOCITY_CONTROL,
    #                             targetVelocity = 0,
    #                             force = 0.0047)

    base_pos, _ = p.getBasePositionAndOrientation(robotId)
    p.resetDebugVisualizerCamera(0.5, 0, -40, base_pos)

    p.setRealTimeSimulation(1)

    for e in range(episodes):
        # reset_joint(robotId, randomize_val=randomize_val, friction= friction)
        reset_joint_swingup(robotId, randomize_val=randomize_val, friction= friction)
        state = get_state(robotId)
        episode_score = 0
        # state = np.array(p.getJointState(robotId, 1)[0], p.getJointState(robotId, 1)[1], p.getJointState(robotId, 2)[0], p.getJointState(robotId, 2)[1])
        eps = max(1-e/decay_steps, 0.05)
        terminal = False
        truncated = False

        for step in count():
            
            prev_state = state.copy()
            state = get_state(robotId)
            # print(f"State: {state}")
            # if state[2] > 0.7 and state[2] < 5.58:
            #     reward = 0
            #     terminal = True
                # print("Terminal due to joint 2")
            if state[0] > 3.14 or state[0] < -3.14:
                reward = -100
                terminal = True
                # print("Terminal due to joint 1")
            else:
                # reward = 1
                reward = math.cos(state[2]) - 0.002 * (state[3]) ** 2 + 0.001 * math.cos(state[0])

            if step > max_steps:
                truncated = True
                # print("Truncated")
            
            if step > 0:
                # print(f"Step: {step}, Prev_state: {prev_state}, Action: {action}, Reward: {reward}, State: {state}, Terminal: {terminal}, Truncated: {truncated}")
                experience = (prev_state, action, reward, state, terminal)
                # print(experience)
                agent.replay_buffer.store(experience)
                episode_score += reward
            
            if terminal or truncated:
                action = -1
                send_action(robotId, action)
                episode_scores.append(episode_score)
                print(f"Episode: {e+1}/{episodes}, Score: {episode_score}")
                if len(agent.replay_buffer) > min_steps_to_learn:
                    print("Learning...")
                    for _ in range(step):
                        agent.learn()
                    print("Learning done")
                if e % update_freq == 0 and e > 1:
                    agent.soft_update_weights()
                    print("Weights updated")
                if (e+1) % save_intervals == 0 and e > 1:
                    save_path = os.path.join(save_dir, f'model_{e+1}.pth')
                    torch.save(agent.online_model, save_path)
                break

                # if e % eval_interval == 0 and eval == True:
                #     eval_score = agent.eval_run()
                #     eval_scores.append(eval_score)
                #     print(f"Eval score: {eval_score}")

            action = agent.choose_action_egreedy(state, eps)
            send_action(robotId, action)
            # print(f"Step: {step}")
            # if step == max_steps:
            #     break
            time.sleep(0.02)

        if e % 50 == 0 and e > 1:
            print(f"Episode: {e}/{episodes}, Rolling mean: {int(np.mean(episode_scores[-50:]))}, Mean: {int(np.mean(episode_scores))}, Epsilon value: {eps}")

        if e % eval_interval == 0 and e > 1:
            eval_score = agent.eval(robotId=robotId, max_steps=max_steps, friction=friction)
            eval_scores.append(eval_score)
            # if len(eval_scores) > 2:
            eval_rolling_mean = int(np.mean(eval_scores[-2:]))
            print(f"Episode: {e}/{episodes}, Eval score: {eval_score}, Eval score rolling mean: {eval_rolling_mean}")
        
        if int(np.mean(episode_scores)) > 200:
            print("Solved")
            break

    p.disconnect()

    plt.plot(episode_scores, label="Episode score")
    window_size = 50
    moving_averages = np.convolve(episode_scores, np.ones(window_size), 'valid') / window_size
    plt.plot(np.arange(window_size, len(episode_scores) + 1), moving_averages, label="50-Episode Moving Average")

    plt.xlabel("Episode")
    plt.ylabel("Episode Score")
    plt.title("DQN CartPole Episode Scores")
    plt.legend()  # Add a legend to the plot
    plt.show()

    # print("Final episode score = ", agent.demo(render=True))
    # agent.env.close()

            



