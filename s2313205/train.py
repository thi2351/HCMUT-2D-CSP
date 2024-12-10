import gym_cutting_stock
import os
import gymnasium as gym
import torch
import numpy as np
from student_submissions.s2313205.A2C_Agent import A2CAgent

# Hyperparameters
NUM_EPISODES = 200
INITIAL_LR = 2e-3
GAMMA = 0.99
MODEL_FILE_PATH = 'a2c_agent_model.pth'
LR_SCHEDULER_GAMMA = 0.99
SAVE_EVERY_N_EPISODES = 2

# Initialize the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the environment
env = gym.make("gym_cutting_stock/CuttingStock-v0", render_mode="human")

# Initialize the A2C agent
agent = A2CAgent(
    device=device,
    num_sheet=100,
    max_w=100,
    max_h=100,
    max_product_type=25,
    max_product_per_type=20,
    hidden_dim=128,
    lr=INITIAL_LR,
    gamma=GAMMA
)

# Load the model if it exists
agent.load_model(MODEL_FILE_PATH)

# Access actor and critic parameters for the optimizer
actor_params = agent.actor.parameters()
critic_params = agent.critic.parameters()

# Combine actor and critic parameters in the optimizer
optimizer = torch.optim.Adam(
    list(actor_params) + list(critic_params),
    lr=INITIAL_LR
)

# Learning rate scheduler with exponential decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_SCHEDULER_GAMMA)

# Training Loop
def train():
    for episode in range(NUM_EPISODES):
        # Reset the environment at the start of each episode
        observation, info = env.reset(seed=None)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Agent selects an action
            action = agent.get_action(observation, info)

            # Take action in the environment and get the new state and reward
            observation, reward, terminated, truncated, info = env.step(action)
            print(info)
            done = terminated or truncated  # Episode ends if either is True
            reward = agent.calculate_reward(observation, done)
            # Track the reward for this episode
            episode_reward += reward
            steps += 1

            # Update the agent (both actor and critic)
            agent.update(done)

        # Print the status of the episode
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f} | Done = {done}")

        # Save the model periodically
        if (episode + 1) % SAVE_EVERY_N_EPISODES == 0:
            agent.save_model(MODEL_FILE_PATH)
            print(f"Model saved after episode {episode + 1}.")

        # Decay learning rate after each episode
        scheduler.step()

    # Close the environment when training is finished
    env.close()

if __name__ == "__main__":
    train()
