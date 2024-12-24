import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count
import torch.optim as optim
import matplotlib.pyplot as plt


class FCQ(nn.Module):                               # Fully connected with at least 2 hidden layers to output Q values for actions based on the state
    def __init__(self, input_dim, output_dim, hidden_dim = (32,32), activation_function = F.relu):
        super(FCQ, self).__init__()

        self.activation_function = activation_function
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dim)-1):
            hidden_layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_layers.append(hidden_layer)


        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)

        if torch.cuda.is_available():               # Use GPU if available
            device = "cuda"
        else:
            device = "cpu"

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):                       # Forward pass through the network. state needs to be a tensor.

        x = self.activation_function(self.input_layer(state))

        for hidden_layer in self.hidden_layers:
            x = self.activation_function(hidden_layer(x))

        x = self.output_layer(x)
        
        return x

class ReplayBuffer():                               # Replay buffer which stores experiences
    def __init__(self, max_size = 50000, batch_size = 64):

        self.max_size = max_size
        self.batch_size = batch_size

        self.states = np.empty(shape=(max_size), dtype=np.ndarray)
        self.actions = np.empty(shape=(max_size), dtype=np.ndarray)
        self.next_states = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rewards = np.empty(shape=(max_size), dtype=np.ndarray)
        self.terminals = np.empty(shape=(max_size), dtype=np.ndarray)

        self.idx = 0
        self.size = 0

    def store(self, experience):                    # Function called after every experience to store it in the replay buffer
        s, a, r, next_s, terminal = experience      # experience contains state, action taken, reward received, next state and if the next state is terminal or not

        self.states[self.idx] = s
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_states[self.idx] = next_s
        self.terminals[self.idx] = terminal

        self.idx += 1
        self.idx = self.idx % self.max_size         # The idx goes back to zero and starts overwriting old experiences when the number of experiences is higher than max_size

        self.size += 1

        self.size = min(self.size, self.max_size)

    def draw_samples(self, batch_size = None):      # This function draws experiences from the replay buffer

        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.states[idxs]), \
                        np.vstack(self.actions[idxs]), \
                        np.vstack(self.rewards[idxs]), \
                        np.vstack(self.next_states[idxs]), \
                        np.vstack(self.terminals[idxs])
        
        return experiences                          # experiences returned as a numpy array

    def __len__(self):
        return self.size

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
        state = torch.tensor(state, dtype=torch.float32, device= self.online_model.device)
        
        # with torch.no_grad():
        #     q = self.online_model.forward(state).cpu().detach()

        # q = q.numpy()
        with torch.no_grad():                               # This is to avoid back-prop
            q = online_model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > eps:
            action = np.argmax(q)
        else:
            action = np.random.randint(self.env.action_space.n)

        return action                                       # action returned as an integer

    def choose_action_greedy(self, state):                  # Chooses action greedily. Used for evaluation of the best policy.
        state = torch.tensor(state, dtype=torch.float32, device=self.online_model.device)
        
        with torch.no_grad():                               # This is to avoid back-prop
            q = self.online_model(state).cpu().detach()

        q = q.numpy()

        action = np.argmax(q)

        return action                                       # action returned as an integer

    def interaction_step(self, state, eps):                 # This function runs one step of interaction with environment and stores the experience
        action = self.choose_action_egreedy(state, eps)
        next_state, reward, terminated, truncated, info = env.step(action)

        experience = (state, action, reward, next_state, terminated)

        self.replay_buffer.store(experience)

        return next_state, reward, terminated, truncated, info

    def learn(self):                                        # This is where the online_model learns and its weights are updated
        states, actions, rewards, next_states, terminals = self.replay_buffer.draw_samples()

        states = torch.tensor(states, dtype=torch.float32, device=self.online_model.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.online_model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.online_model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.online_model.device)
        terminals = torch.tensor(terminals, dtype=torch.float32, device=self.online_model.device)

        # qsa_next = self.target_model(next_states).detach()
        qsa_next_max = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        yj = rewards + (self.gamma * qsa_next_max * (1 - terminals))

        qsa = self.online_model(states).gather(1,actions)

        td_error = yj - qsa
        loss = td_error.pow(2).mul(0.5).mean()
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
            

env = gym.make('CartPole-v1', render_mode = "rgb_array")

epochs = 40
episodes = 2000
max_steps = 1000

learning_rate = 0.0005
batch_size = 64
warmup_batches = 5
update_freq = 10                                        # Number signifies episodes after which target network will be updated
min_steps_to_learn = warmup_batches * batch_size

replay_buffer = ReplayBuffer(batch_size=batch_size)
online_model = FCQ(env.observation_space.shape[0], env.action_space.n, (512,128))
target_model = FCQ(env.observation_space.shape[0], env.action_space.n, (512,128))
optimizer = optim.RMSprop(online_model.parameters(), lr=learning_rate)

agent = DQN(env, replay_buffer, online_model, target_model, optimizer, warmup_batches, update_freq, epochs=epochs)
agent.soft_update_weights()
episode_scores = []
eval_scores = []
decay_steps = 2000
eval = True

for e in range(episodes+1):
    state, _ = env.reset()
    episode_score = 0
    eps = max(1-e/decay_steps, 0.05)

    for step in count():
        state, reward, terminated, truncated, _ = agent.interaction_step(state, eps)
        episode_score += reward                             # Score with epsilon greedy action selection (with exploration)

        if terminated or truncated or step > max_steps:
            if len(agent.replay_buffer) > min_steps_to_learn:
                agent.learn()
            if e % update_freq == 0 and step > 1:
                agent.soft_update_weights()
            break
    
    episode_scores.append(episode_score)
    
    if eval:
        eval_score = agent.demo(render=False)               # Eval score is calculated using agent.demo which uses greedy action selection
        eval_scores.append(eval_score)

    if e % 50 == 0 and e > 1:
        print("Episode: {}/{}, Rolling mean: {}, Mean: {}, Epsilon value: {:.2f}".format(e, episodes, int(np.mean(episode_scores[-50:])), int(np.mean(episode_scores)), eps))

    if np.mean(eval_scores[-50:]) > 300:
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
plt.ylabel("Evaluation Score")
plt.title("DQN CartPole-v1 Evaluation Scores")
plt.legend()  # Add a legend to the plot
plt.show()

print("Final eval score = ", agent.demo(render=True))
agent.env.close()