import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from policy import Policy  # Ensure this is correctly defined elsewhere
import numpy as np

class StateEncoder(nn.Module):
    def __init__(self, num_sheet, max_w, max_h, max_product_type, max_product_per_type):
        super(StateEncoder, self).__init__()
        self.num_sheet = num_sheet
        self.max_w = max_w
        self.max_h = max_h
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type

        # CNN layers with 1 input and 1 output channel
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/2, max_h/2)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/4, max_h/4)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/8, max_h/8)

        self.flatten = nn.Flatten()

    def preprocess(self, observation_stocks):
        """
        Convert stocks to binary tensors where 1.0 represents empty cells (-1) and 0.0 otherwise.
        """
        # Convert list of NumPy arrays to a single NumPy array
        stocks_np = np.stack(observation_stocks)  # Shape: (num_sheet, max_w, max_h)
        # Preprocess: 1.0 for empty (-1), 0.0 otherwise
        stocks_binary = (stocks_np == -1).astype(np.float32)
        # Convert to torch tensor and add channel dimension
        stocks_tensor = torch.from_numpy(stocks_binary).unsqueeze(1)  # Shape: (num_sheet, 1, max_w, max_h)
        return stocks_tensor

    def forward(self, observation_stocks, observation_products):
        # Preprocess stocks
        stocks_tensor = self.preprocess(observation_stocks).to(next(self.parameters()).device)
        # Pass through CNN
        cnn_outs = []
        for i in range(self.num_sheet):
            sheet = stocks_tensor[i].unsqueeze(0)  # Shape: (1, 1, W, H)
            sheet = F.relu(self.conv1(sheet))      # Shape: (1, 1, W/2, H/2)
            sheet = F.relu(self.conv2(sheet))      # Shape: (1, 1, W/4, H/4)
            sheet = F.relu(self.conv3(sheet))      # Shape: (1, 1, W/8, H/8)
            sheet = self.flatten(sheet)             # Shape: (1, (W/8) * (H/8))
            cnn_outs.append(sheet)
        # Concatenate all sheet features
        cnn_out = torch.cat(cnn_outs, dim=1)  # Shape: (1, num_sheet * (W/8) * (H/8))

        # Process product features
        product_features = []
        for product in observation_products:
            length, width = product["size"]
            quantity = product["quantity"]
            # Consider both orientations
            product_features.append([length, width, quantity])
            product_features.append([width, length, quantity])
        # If fewer than 2 * max_product_type, pad with zeros
        while len(product_features) < 2 * self.max_product_type:
            product_features.append([0.0, 0.0, 0.0])
        product_features = np.array(product_features[:2 * self.max_product_type], dtype=np.float32)
        product_features = torch.from_numpy(product_features).view(1, -1).to(next(self.parameters()).device)  # Shape: (1, 6 * max_product_type)

        # Combine
        combined_input = torch.cat([cnn_out, product_features], dim=1)  # Shape: (1, num_sheet * (W/8) * (H/8) + 6 * max_product_type)
        return combined_input


class ActorNetwork(nn.Module):
    def __init__(self, num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.num_sheet = num_sheet
        self.max_w = max_w
        self.max_h = max_h
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = StateEncoder(num_sheet, max_w, max_h, max_product_type, max_product_per_type)

        # Calculate CNN output dimension
        # After conv1, conv2, conv3: each reduces spatial dimensions by factor of 2
        # So final spatial size: max_w / 8, max_h / 8
        # Since channels are 1, cnn_output_dim = num_sheet * (max_w//8) * (max_h//8)
        cnn_output_dim = num_sheet * (max_w // 8 + 1) * (max_h // 8 + 1)
        product_output_dim = 6 * max_product_type
        input_dim = cnn_output_dim + product_output_dim

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_sheet * max_product_type)

    def get_valid_actions(self, observation_products):
        """
        Xác định các hành động hợp lệ dựa trên số lượng sản phẩm và quantity.
        """
        num_products = len(observation_products)
        valid_actions = torch.zeros(self.num_sheet * self.max_product_type, dtype=torch.bool).to(next(self.parameters()).device)
        for sheet in range(self.num_sheet):
            for product in range(num_products):
                if product < self.max_product_type:
                    product_info = observation_products[product]
                    if product_info["quantity"] > 0:
                        action_index = sheet * self.max_product_type + product
                        valid_actions[action_index] = True
        return valid_actions

    def forward(self, observation_stocks, observation_products):
        combined_input = self.encoder(observation_stocks, observation_products)
        x = F.relu(self.fc1(combined_input))
        action_logits = self.fc2(x)  # Shape: (1, num_sheet * max_product_type)

        # Get valid actions mask
        valid_actions = self.get_valid_actions(observation_products)  # Shape: (num_sheet * max_product_type,)

        # Expand mask to match batch size if necessary
        if action_logits.dim() == 2:
            valid_actions = valid_actions.unsqueeze(0).expand(action_logits.size(0), -1)  # Shape: (batch_size, num_sheet * max_product_type)

        # Set logits of invalid actions to a very large negative value
        action_logits[~valid_actions] = -1e9

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

        # Shared encoder
        self.encoder = StateEncoder(num_sheet, max_w, max_h, max_product_type, max_product_per_type)

        # Calculate CNN output dimension
        cnn_output_dim = num_sheet * (max_w // 8 + 1) * (max_h // 8 + 1)
        product_output_dim = 6 * max_product_type
        input_dim = cnn_output_dim + product_output_dim

        # Fully connected layers for value
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, observation_stocks, observation_products):
        combined_input = self.encoder(observation_stocks, observation_products)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(1)  # Return as (batch_size,)


class A2CAgent(Policy):
    def __init__(self, device, num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim, 
                 lr=0.001, gamma=0.9):
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

        self.actor = ActorNetwork(num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim).to(self.device)
        self.critic = CriticNetwork(num_sheet, max_w, max_h, max_product_type, max_product_per_type, hidden_dim).to(self.device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        # To store trajectories
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def get_valid_actions(self, observation_products):
        """
        Xác định các hành động hợp lệ dựa trên số lượng sản phẩm và quantity.
        """
        num_products = len(observation_products)
        valid_actions = torch.zeros(self.num_sheet * self.max_product_type, dtype=torch.bool).to(self.device)
        for sheet in range(self.num_sheet):
            for product in range(num_products):
                if product < self.max_product_type:
                    product_info = observation_products[product]
                    if product_info["quantity"] > 0:
                        action_index = sheet * self.max_product_type + product
                        valid_actions[action_index] = True
        return valid_actions

    def selectSheet(self, observation):
        """
        Chọn một hành động dựa trên phân phối xác suất của mạng actor.
        """
        # Extract stocks and products
        observation_stocks = observation["stocks"]    
        observation_products = observation["products"]

        # Move data to device and convert to tensors
        action_probs = self.actor(observation_stocks, observation_products).to(self.device)  # Shape: (1, num_sheet * max_product_type)
        distribution = torch.distributions.Categorical(action_probs)
        selected_action = distribution.sample()
        log_prob = distribution.log_prob(selected_action)
        entropy = distribution.entropy()

            # Store for training
        self.log_probs.append(log_prob)
        self.values.append(self.critic(observation_stocks, observation_products))
        self.entropies.append(entropy)

        return selected_action.item()

    def place_product(self, observation, sheet_index, product_index):
        """
        Tìm vị trí tốt nhất để đặt sản phẩm trên sheet được chọn.
        """
        stock = observation["stocks"][sheet_index]
        prod_info = observation["products"][product_index]
        (prod_w, prod_h) = prod_info["size"]
        max_w, max_h = stock.shape

        best_pos_x, best_pos_y = None, None

        for pos_x in range(max_w - prod_w + 1):
            chosen_pos_y = None
            for pos_y in range(max_h - prod_h + 1):
                if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                    chosen_pos_y = pos_y
                    break  
            if chosen_pos_y is not None:
                if best_pos_y is None:
                    best_pos_x, best_pos_y = pos_x, chosen_pos_y
                else:
                    if (chosen_pos_y < best_pos_y) or (chosen_pos_y == best_pos_y and pos_x < best_pos_x):
                        best_pos_x, best_pos_y = pos_x, chosen_pos_y
        if best_pos_y is None:
            return None
        return {"stock_idx": sheet_index, 
               "size": [prod_w, prod_h],
               "position": [best_pos_x, best_pos_y]}

    def get_action(self, observation, info): 
        """
        Chọn hành động dựa trên trạng thái hiện tại và xử lý các hành động không hợp lệ.
        """
        selected_component = self.selectSheet(observation)
        sheet_index = selected_component // self.max_product_type
        product_index = selected_component % self.max_product_type

        # Kiểm tra tính hợp lệ của product_index
        if product_index >= len(observation["products"]):
            # Hành động không hợp lệ: product_index ngoài phạm vi
            self.store_reward(-1.0)  # Phần thưởng tiêu cực
            self.update()  # Cập nhật mạng neuron ngay lập tức
            return None
        else:
            product_info = observation["products"][product_index]
            if product_info["quantity"] <= 0:
                # Hành động không hợp lệ: quantity = 0
                self.store_reward(-1.0)  # Phần thưởng tiêu cực
                self.update()  # Cập nhật mạng neuron ngay lập tức
                return None

            selected_action = self.place_product(observation, sheet_index, product_index)
            if selected_action is not None:
                # Hành động hợp lệ
                self.store_reward(1.0)  # Phần thưởng tích cực
                return selected_action
            else:
                # Không thể đặt sản phẩm: có thể đã chọn sai vị trí
                self.store_reward(-1.0)  # Phần thưởng tiêu cực
                self.update()  # Cập nhật mạng neuron ngay lập tức
                return None

    def update(self):
        """
        Cập nhật mạng neuron dựa trên các trải nghiệm đã lưu.
        """
        if len(self.rewards) == 0:
            return  # Không có gì để cập nhật

        # Compute returns
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        values = torch.stack(self.values)  # Shape: (batch_size,)
        log_probs = torch.stack(self.log_probs)  # Shape: (batch_size,)
        entropies = torch.stack(self.entropies)  # Shape: (batch_size,)

        advantage = returns - values  # Shape: (batch_size,)

        # Actor loss
        actor_loss = -(log_probs * advantage.detach()).mean() - 0.01 * entropies.mean()

        # Critic loss
        critic_loss = advantage.pow(2).mean()

        # Backpropagation
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # Clear trajectories
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def store_reward(self, reward):
        """
        Lưu trữ phần thưởng để sử dụng trong quá trình cập nhật.
        """
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))

    def _can_place_(self, stock, position, size):
        """
        Kiểm tra xem sản phẩm có thể được đặt tại vị trí cụ thể trên stock hay không.
        """
        x, y = position
        w, h = size
        return np.all(stock[x:x+w, y:y+h] == -1)

    def close(self):
        """
        Thực hiện bất kỳ công việc dọn dẹp nào cần thiết khi đóng agent.
        """
        pass  # Implement any cleanup if necessary
