import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count

import torch.optim as optim

class FCQ(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = (32,32), activation_function = F.relu):
        super(FCQ, self).__init__()

        self.activation_function = activation_function
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dim)-1):
            hidden_layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_layers.append(hidden_layer)


        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):       # state needs to be a tensor

        x = self.activation_function(self.input_layer(state))

        for hidden_layer in self.hidden_layers:
            x = self.activation_function(hidden_layer(x))

        x = self.output_layer(x)
        
        return x

class ReplayBuffer():
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

    def store(self, experience):
        s, a, r, next_s, terminal = experience

        self.states[self.idx] = s
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_states[self.idx] = next_s
        self.terminals[self.idx] = terminal

        self.idx += 1
        self.idx = self.idx % self.max_size

        self.size += 1

        self.size = min(self.size, self.max_size)

    def draw_samples(self, batch_size = None):

        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.states[idxs]), \
                        np.vstack(self.actions[idxs]), \
                        np.vstack(self.rewards[idxs]), \
                        np.vstack(self.next_states[idxs]), \
                        np.vstack(self.terminals[idxs])
        
        return experiences      # experiences returned as a numpy array

    def __len__(self):
        return self.size

class DQN():
    def __init__(self, env, replay_buffer, online_model, target_model, optimizer, warmup_batches, update_freq, epochs = 40, gamma = 1):
        self.env = env
        self.replay_buffer = replay_buffer
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.warmup_batches = warmup_batches
        self.update_freq = update_freq
        self.gamma = gamma
        self.demo_env = gym.make("CartPole-v1", render_mode = "human")

    def choose_action_egreedy(self, state, eps):
        state = torch.tensor(state, dtype=torch.float32, device= self.online_model.device)
        
        # with torch.no_grad():
        #     q = self.online_model.forward(state).cpu().detach()

        # q = q.numpy()
        with torch.no_grad():
            q = online_model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > eps:
            action = np.argmax(q)
        else:
            action = np.random.randint(self.env.action_space.n)

        return action           # action returned as an integer

    def choose_action_greedy(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.online_model.device)
        
        with torch.no_grad():
            q = self.online_model(state).cpu().detach()

        q = q.numpy()

        action = np.argmax(q)

        return action           # action returned as an integer

    def interaction_step(self, state, eps):
        action = self.choose_action_egreedy(state, eps)
        next_state, reward, terminated, truncated, info = env.step(action)

        experience = (state, action, reward, next_state, terminated)

        self.replay_buffer.store(experience)

        return next_state, reward, terminated, truncated, info

    def learn(self):
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

    def demo(self):
        state, _ = self.demo_env.reset()
        final_demo_score = 0
        for step in count():
            action = self.choose_action_greedy(state)
            state, reward, terminated, truncated, info = self.demo_env.step(action)
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
update_freq = 10                    # Number signifies episodes after which target network will be updated
min_steps_to_learn = warmup_batches * batch_size

replay_buffer = ReplayBuffer(batch_size=batch_size)
online_model = FCQ(env.observation_space.shape[0], env.action_space.n, (512,128))
target_model = FCQ(env.observation_space.shape[0], env.action_space.n, (512,128))
optimizer = optim.RMSprop(online_model.parameters(), lr=learning_rate)

agent = DQN(env, replay_buffer, online_model, target_model, optimizer, warmup_batches, update_freq, epochs=epochs)
agent.soft_update_weights()
episode_scores = []
decay_steps = 2000

for e in range(episodes+1):
    state, _ = env.reset()
    episode_score = 0
    eps = max(1-e/decay_steps, 0.05)

    for step in count():
        state, reward, terminated, truncated, _ = agent.interaction_step(state, eps)
        episode_score += reward


        if len(agent.replay_buffer) > min_steps_to_learn:
            agent.learn()

        if step % update_freq == 0 and step > 1:
            agent.soft_update_weights()

        if terminated or truncated or step > max_steps:
            break
    
    episode_scores.append(episode_score)
    

    if e % 50 == 0 and e > 1:
        print("Episode: {}/{}, Rolling mean: {}, Mean: {}, Epsilon value: {:.2f}".format(e, episodes, int(np.mean(episode_scores[-50:])), int(np.mean(episode_scores)), eps))

    if np.mean(episode_scores[-100:]) > 200:
        print("Episode: {}/{}, Rolling mean: {}, Mean: {}, Epsilon value: {:.2f}".format(e, episodes, int(np.mean(episode_scores[-50:])), int(np.mean(episode_scores)), eps))
        print("Cartpole solved!")
        break

agent.env.close()


print("Final eval score = ", agent.demo())
agent.env.close()