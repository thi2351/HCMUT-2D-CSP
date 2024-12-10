import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from policy import Policy  # Ensure that the Policy class is properly defined
import os

class StateEncoder(nn.Module):
    def __init__(self, num_sheet, max_w, max_h, max_product_type, max_product_per_type):
        super(StateEncoder, self).__init__()
        self.num_sheet = num_sheet
        self.max_w = max_w
        self.max_h = max_h
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type

        # CNN layers to extract features
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/2, max_h/2)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/4, max_h/4)

        self.flatten = nn.Flatten()

    def preprocess(self, observation_stocks):
        """
        Convert stock sheets into binary tensor: 1.0 for empty cells (-1), 0.0 for filled cells.
        """
        stocks_np = np.stack(observation_stocks)  # Shape: (num_sheet, max_w, max_h)
        stocks_binary = (stocks_np == -1).astype(np.float32)
        stocks_tensor = torch.from_numpy(stocks_binary).unsqueeze(1)  # Shape: (num_sheet, 1, max_w, max_h)
        return stocks_tensor

    def forward(self, observation_stocks, observation_products):
        stocks_tensor = self.preprocess(observation_stocks).to(next(self.parameters()).device)
        cnn_outs = []
        for i in range(self.num_sheet):
            sheet = stocks_tensor[i].unsqueeze(0)  # Shape: (1, 1, W, H)
            sheet = F.relu(self.conv1(sheet))      # Shape: (1, 1, W/2, H/2)
            sheet = F.relu(self.conv2(sheet))      # Shape: (1, 1, W/4, H/4)
            sheet = self.flatten(sheet)            # Shape: (1, W/4 * H/4)
            cnn_outs.append(sheet)
        cnn_out = torch.cat(cnn_outs, dim=1)  # Shape: (1, num_sheet * (W/4 * H/4))

        product_features = []
        for product in observation_products:
            length, width = product["size"]
            quantity = product["quantity"]
            product_features.append([length, width, quantity])
        while len(product_features) < self.max_product_type:
            product_features.append([0.0, 0.0, 0.0])
        product_features = product_features[:self.max_product_type]

    # Convert to numpy array and then to torch tensor
        product_features = np.array(product_features, dtype=np.float32)
        product_features = torch.from_numpy(product_features).view(1, -1).to(next(self.parameters()).device) 
        combined_input = torch.cat([cnn_out, product_features], dim=1)  # Shape: (1, cnn_dim + product_dim)
        return combined_input

class ActorNetwork(nn.Module, Policy):
    def __init__(self, num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.num_sheet = num_sheet
        self.max_w = max_w
        self.max_h = max_h
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.hidden_dim = hidden_dim

        self.encoder = StateEncoder(num_sheet, max_w, max_h, max_product_type, max_product_per_type)

        cnn_output_dim = 4 * num_sheet * (max_w // 4) * (max_h // 4)  # Adjust size calculation
        product_output_dim = 3 * max_product_type
        input_dim = cnn_output_dim + product_output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * num_sheet * max_product_type) 
    def _can_place_(self, stock, position, size):
        """
        Checks if a product can be placed at a specific position on the stock.
        """
        x, y = position
        w, h = size

        # Convert stock to a torch.Tensor if it's a numpy array
        if isinstance(stock, np.ndarray):
            stock = torch.tensor(stock, dtype=torch.int32)

        # Perform the check using torch.all
        return torch.all(stock[x:x + w, y:y + h] == -1)

    def get_valid_actions(self, observation_products, observation_stocks):
        """
        Returns a tensor where valid actions (where products can be placed or rotated) are marked as True.
        """
        num_products = len(observation_products)
        valid_actions = torch.zeros(2 * self.num_sheet * self.max_product_type, dtype=torch.bool).to(next(self.parameters()).device)

        for sheet in range(self.num_sheet):
            for product_idx in range(num_products):
                if product_idx < self.max_product_type and observation_products[product_idx]["quantity"] > 0:
                    # Action index with rotation (m = 0 or m = 1)
                    action_index_base = sheet * self.max_product_type + product_idx
                    for m in range(2):  # 0 -> No rotation, 1 -> Rotation
                        action_index = action_index_base * 2 + m
                        product_info = observation_products[product_idx]
                        prod_w, prod_h = product_info["size"]

                        # Try placing the product without rotation
                        if m == 0:
                            for i in range(self.max_w - prod_w + 1):
                                for j in range(self.max_h - prod_h + 1):
                                    if self._can_place_(observation_stocks[sheet], (i, j), (prod_w, prod_h)):
                                        valid_actions[action_index] = True
                                        break
                            if valid_actions[action_index]:
                                break  # No need to check further if valid action found

                        # Try placing the product with rotation
                        elif m == 1:
                            prod_w, prod_h = prod_h, prod_w  # Rotate dimensions
                            for i in range(self.max_w - prod_w + 1):
                                for j in range(self.max_h - prod_h + 1):
                                    if self._can_place_(observation_stocks[sheet], (i, j), (prod_w, prod_h)):
                                        valid_actions[action_index] = True
                                        break
                            if valid_actions[action_index]:
                                break

        return valid_actions


    def forward(self, observation_stocks, observation_products):
        combined_input = self.encoder(observation_stocks, observation_products)
        x = F.relu(self.fc1(combined_input))
        action_logits = self.fc2(x)  # Shape: (1, num_sheet * max_product_type)

        valid_actions = self.get_valid_actions(observation_products, observation_stocks)

        if action_logits.dim() == 2:
            valid_actions = valid_actions.unsqueeze(0).expand(action_logits.size(0), -1)

        action_logits[~valid_actions] = -1e9  # Mask invalid actions
        action_probs = F.softmax(action_logits, dim=1)

        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.num_sheet = num_sheet
        self.max_w = max_w
        self.max_h = max_h
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.hidden_dim = hidden_dim

        self.encoder = StateEncoder(num_sheet, max_w, max_h, max_product_type, max_product_per_type)

        cnn_output_dim = 4 * num_sheet * (max_w // 4) * (max_h // 4)
        product_output_dim = 3 * max_product_type
        input_dim = cnn_output_dim + product_output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, observation_stocks, observation_products):
        combined_input = self.encoder(observation_stocks, observation_products)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        # Debugging log
        print(f"Critic Value: {value}")

        return value.squeeze(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from policy import Policy  # Ensure that the Policy class is properly defined
import os

class A2CAgent(Policy):
    def __init__(self, device, num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim, lr=0.001, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        super(A2CAgent, self).__init__()
        self.device = device
        self.num_sheet = num_sheet
        self.max_w = max_w
        self.max_h = max_h
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon  # Epsilon for exploration-exploitation trade-off
        self.epsilon_min = epsilon_min  # Minimum value for epsilon
        self.epsilon_decay = epsilon_decay  # Epsilon decay factor

        self.actor = ActorNetwork(num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim).to(self.device)
        self.critic = CriticNetwork(num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim).to(self.device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def select_action(self, observation):
        observation_stocks = observation["stocks"]
        observation_products = observation["products"]

        # Get action probabilities from the actor network
        action_probs = self.actor(observation_stocks, observation_products)
        distribution = torch.distributions.Categorical(action_probs)
        
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action from the valid actions
            selected_action = distribution.sample()  # Random action based on the policy
        else:
            # Exploitation: choose the best action (greedy approach)
            selected_action = torch.argmax(action_probs, dim=1)  # Best action based on current policy

        log_prob = distribution.log_prob(selected_action)
        entropy = distribution.entropy()

        # Store for A2C updates
        self.log_probs.append(log_prob)
        self.values.append(self.critic(observation_stocks, observation_products))
        self.entropies.append(entropy)

        # Decode action to (sheet_idx, product_idx, rotate)
        action_index = selected_action.item()
        sheet_index = action_index // (2 * self.max_product_type)
        product_index = (action_index % (2 * self.max_product_type)) // 2
        rotate = action_index % 2  # 0 for no rotation, 1 for rotation

        return sheet_index, product_index, rotate

    def place_product(self, observation, sheet_index, product_index, rotate):
        stock = observation["stocks"][sheet_index]
        prod_info = observation["products"][product_index]
        prod_w, prod_h = prod_info["size"]
        if rotate == 1:
            prod_w, prod_h = prod_h, prod_w  # Rotate dimensions

        max_w, max_h = stock.shape
        best_pos_x, best_pos_y = None, None

        for pos_x in range(max_w - prod_w + 1):
            for pos_y in range(max_h - prod_h + 1):
                if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                    best_pos_x, best_pos_y = pos_x, pos_y
                    break
            if best_pos_x is not None:
                break

        if best_pos_y is None:
            return None

        return {
            "stock_idx": sheet_index, 
            "size": [prod_w, prod_h],
            "position": [best_pos_x, best_pos_y],
        }

    def get_action(self, observation, info):
        selected_action = self.select_action(observation)
        sheet_idx, product_idx, rotate = selected_action 
        placed = self.place_product(observation, sheet_idx, product_idx, rotate)
        return placed

    def update(self, done):
        if len(self.rewards) == 0:
            return

        assert len(self.rewards) == len(self.log_probs) == len(self.values) == len(self.entropies)
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).squeeze(1).to(self.device)

        returns_mean = returns.mean()
        returns_std = returns.std(unbiased=False)
        returns = (returns - returns_mean) / (returns_std + 1e-5)

        values = torch.stack(self.values).squeeze(1).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        entropies = torch.stack(self.entropies).to(self.device)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean() - 0.1 * entropies.mean()
        critic_loss = advantage.pow(2).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def store_reward(self, reward):
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))

    def calculate_reward(self, observation, done, C=1):
        waste_total = 0
        area_total = 0
        stock_count = 0
        if not done:
            print(f"Reward: 0")
            return 0
        for idx, sheet in enumerate(observation["stocks"]):
            if np.any(sheet >= 0):
                waste = np.sum(sheet == -1)
                area = np.sum(sheet != -2)
                waste_total += waste
                area_total += area
                stock_count += 1

        waste_ratio = waste_total / area_total if area_total > 0 else 0

        reward = C / waste_ratio  # Penalty for waste

        print(f"Reward: {reward}")
        return reward

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        return np.all(stock[x:x + w, y:y + h] == -1)

    def save_model(self, filepath):
        torch.save(self.actor.state_dict(), filepath + '_actor.pth')
        torch.save(self.critic.state_dict(), filepath + '_critic.pth')
        print(f"Actor and Critic models saved to {filepath}_actor.pth and {filepath}_critic.pth.")

    def load_model(self, filepath, train=True):
        actor_path = filepath + '_actor.pth'
        critic_path = filepath + '_critic.pth'

        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            print(f"Error: Model files not found. Initializing model from scratch.")
            self.initialize_model()
            return

        try:
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"Models loaded from {actor_path} and {critic_path}.")
            if not train:
                self.actor.eval()
                self.critic.eval()
            else:
                self.actor.train()
                self.critic.train()

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.initialize_model()

    def initialize_model(self):
        print("Initializing Actor and Critic networks...")
        self.actor = ActorNetwork(self.num_sheet, self.max_w, self.max_h, self.max_product_type, self.max_product_per_type, self.hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.num_sheet, self.max_w, self.max_h, self.max_product_type, self.max_product_per_type, self.hidden_dim).to(self.device)

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.actor.train()
        self.critic.train()

        print("Actor and Critic networks initialized from scratch. Starting training...")

    def close(self):
        pass
