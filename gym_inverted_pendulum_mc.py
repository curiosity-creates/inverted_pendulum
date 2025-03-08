import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count
import torch.optim as optim
import matplotlib.pyplot as plt
from gym_inverted_pendulum import FCQ               # Importing FCQ class from gym_inverted_pendulum
from gym_inverted_pendulum import ReplayBuffer      # Importing ReplayBuffer class from gym_inverted_pendulum


class DQN():                                        # Reinforcement learning agent
    def __init__(self, env, replay_buffer, online_model, target_model, optimizer, warmup_batches, update_freq, epochs = 40, gamma = 1, demo_env=None):
        self.env = env
        self.replay_buffer = replay_buffer
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.warmup_batches = warmup_batches
        self.update_freq = update_freq
        self.gamma = gamma
        if demo_env is None:
            self.demo_env = gym.make("CartPole-v1", render_mode = "human")
        else:
            self.demo_env = demo_env

    def choose_action_egreedy(self, state, eps):            # Chooses action epsilon greedily. Helps exploration.        
        # Converting state to a tensor and pushing the tensor to GPU
        state = torch.tensor(state, dtype=torch.float32, device= self.online_model.device)
        
        with torch.no_grad():                               # This is to avoid back-prop
            q = online_model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > eps:                          # Random action taken with probability epsilon
            action = np.argmax(q)
        else:                                               # Maximizing action taken with probability (1-eps)
            action = np.random.randint(self.env.action_space.n)

        return action                                       # action returned as an integer

    def choose_action_greedy(self, state):                  # Chooses action greedily. Used for evaluation of the best policy.
        # Converting state to a tensor and pushing the tensor to GPU
        state = torch.tensor(state, dtype=torch.float32, device=self.online_model.device)
        
        with torch.no_grad():                               # This is to avoid back-prop
            q = self.online_model(state).cpu().detach()

        q = q.numpy()

        action = np.argmax(q)                               # Maximizing action always

        return action                                       # action returned as an integer

    def interaction_step(self, state, eps):                 # This function runs one step of interaction with environment and stores the experience
        action = self.choose_action_egreedy(state, eps)
        next_state, reward, terminated, truncated, info = env.step(action)

        experience = (state, action, reward, next_state, terminated)

        self.replay_buffer.store(experience)                # Storing experience in the replay buffer

        return next_state, reward, terminated, truncated, info

    def learn(self):                                        # This is where the online_model learns and its weights are updated
        states, actions, rewards, next_states, terminals = self.replay_buffer.draw_samples()

        # Converting numpy arrays to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.online_model.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.online_model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.online_model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.online_model.device)
        terminals = torch.tensor(terminals, dtype=torch.float32, device=self.online_model.device)

        # Calculating the target. Notice the use of target model in the line below and the use of detach to avoid back-prop
        qsa_next_max = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        yj = rewards + (self.gamma * qsa_next_max * (1 - terminals))    # (1-terminals) is to avoid using terminal experiences when calculating the target.

        qsa = self.online_model(states).gather(1,actions)   # Notice the use of online model here. We want to back-prop. So detach not used.

        td_error = yj - qsa
        loss = td_error.pow(2).mul(0.5).mean()              # MSE loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        


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

    def demo(self, render=False):                       # Runs a full episode but with greedy actions. The episode can be rendered if render=True
        if render:
            env = self.demo_env
        else:
            env = self.env
        state, _ = env.reset()
        final_demo_score = 0
        for step in count():
            action = self.choose_action_greedy(state)
            state, reward, terminated, truncated, info = env.step(action)
            final_demo_score += reward
            
            if terminated or truncated or step > 500:
                break
        
        return final_demo_score
            
if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode = "rgb_array")

    epochs = 40
    episodes = 2000
    max_steps = 1000

    learning_rate = 0.0005                                  # Alpha
    batch_size = 256
    warmup_batches = 5
    update_freq = 50                                        # Number signifies episodes after which target network will be updated
    min_steps_to_learn = warmup_batches * batch_size

    replay_buffer = ReplayBuffer(batch_size=batch_size)

    # Create online and target models, input dimension is number of variables describing the state, output dimensions would number of actions possible.
    online_model = FCQ(env.observation_space.shape[0], env.action_space.n, (1024,128))
    target_model = FCQ(env.observation_space.shape[0], env.action_space.n, (1024,128))
    optimizer = optim.RMSprop(online_model.parameters(), lr=learning_rate)

    # Creating an instance of the DQN class.
    agent = DQN(env, replay_buffer, online_model, target_model, optimizer, warmup_batches, update_freq, epochs=epochs)
    agent.soft_update_weights()
    episode_scores = []
    eval_scores = []
    decay_steps = episodes * 0.8
    eval = True

    for e in range(episodes+1):                                 # Training begins
        state, _ = env.reset()                                  # Environment reset at the beginning of every episode
        episode_score = 0
        eps = max(1-e/decay_steps, 0.05)                        # This allows epsilon decay

        for step in count():
            state, reward, terminated, truncated, _ = agent.interaction_step(state, eps)    # Agent interaction with the environment
            episode_score += reward                             # Score with epsilon greedy action selection (with exploration)

            # Learning happens only at the end of the episode and not after every time step
            if terminated or truncated or step > max_steps:
                if len(agent.replay_buffer) > min_steps_to_learn:
                    # for _ in range(epochs):
                    agent.learn()
                if e % update_freq == 0 and step > 1:           # This makes the target network and omline network the same after a certain number of steps
                    agent.soft_update_weights()
                break
        
        episode_scores.append(episode_score)
        
        if eval:
            eval_score = agent.demo(render=False)               # Eval score is calculated using agent.demo which uses greedy action selection
            eval_scores.append(eval_score)

        if e % 50 == 0 and e > 1:
            print("Episode: {}/{}, Rolling mean: {}, Mean: {}, Epsilon value: {:.2f}".format(e, episodes, int(np.mean(episode_scores[-50:])), int(np.mean(episode_scores)), eps))

        if np.mean(eval_scores[-50:]) > 300:                    # Moving average of episode returns
            print("Episode: {}/{}, Rolling mean: {}, Mean: {}, Epsilon value: {:.2f}".format(e, episodes, int(np.mean(episode_scores[-50:])), int(np.mean(episode_scores)), eps))
            print("Eval score for the last 50 episodes: ", np.mean(eval_scores[-50:]))
            print("Cartpole solved!")
            break

    agent.env.close()

    # Plotting the eval_scores
    plt.plot(eval_scores, label="Evaluation score")
    window_size = 50
    moving_averages = np.convolve(eval_scores, np.ones(window_size), 'valid') / window_size
    plt.plot(np.arange(window_size, len(eval_scores) + 1), moving_averages, label="50-Episode Moving Average")

    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("DQN CartPole-v1 Evaluation Scores")
    plt.legend()  # Add a legend to the plot
    plt.show()

    print("Final eval score = ", agent.demo(render=True))
    agent.env.close()