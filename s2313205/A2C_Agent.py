import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from policy import Policy  # Đảm bảo Policy được định nghĩa đúng ở nơi khác

class StateEncoder(nn.Module):
    def __init__(self, num_sheet, max_w, max_h, max_product_type, max_product_per_type):
        super(StateEncoder, self).__init__()
        self.num_sheet = num_sheet
        self.max_w = max_w
        self.max_h = max_h
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type

        # CNN layers với số kênh tăng dần để trích xuất đặc trưng tốt hơn
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/2, max_h/2)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/4, max_h/4)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # Output: (1, max_w/8, max_h/8)

        self.flatten = nn.Flatten()

    def preprocess(self, observation_stocks):
        """
        Chuyển đổi các tấm stock thành tensor nhị phân:
        1.0 cho các ô trống (-1) và 0.0 cho các ô đã đặt vật phẩm.
        """
        # Chuyển đổi danh sách các mảng NumPy thành một mảng NumPy duy nhất
        stocks_np = np.stack(observation_stocks)  # Shape: (num_sheet, max_w, max_h)
        # Tiền xử lý: 1.0 cho các ô trống (-1), 0.0 cho các ô đã đặt
        stocks_binary = (stocks_np == -1).astype(np.float32)
        # Chuyển đổi thành tensor PyTorch và thêm kênh
        stocks_tensor = torch.from_numpy(stocks_binary).unsqueeze(1)  # Shape: (num_sheet, 1, max_w, max_h)
        return stocks_tensor

    def forward(self, observation_stocks, observation_products):
        # Tiền xử lý các tấm stock
        stocks_tensor = self.preprocess(observation_stocks).to(next(self.parameters()).device)
        # Qua các lớp CNN
        cnn_outs = []
        for i in range(self.num_sheet):
            sheet = stocks_tensor[i].unsqueeze(0)  # Shape: (1, 1, W, H)
            sheet = F.relu(self.conv1(sheet))      # Shape: (1, 1, W/2, H/2)
            sheet = F.relu(self.conv2(sheet))      # Shape: (1, 1, W/4, H/4)
            sheet = F.relu(self.conv3(sheet))      # Shape: (1, 1, W/8, H/8)
            sheet = self.flatten(sheet)             # Shape: (1, 1 * (W/8) * (H/8))
            cnn_outs.append(sheet)
        # Nối các đặc trưng của tất cả các tấm stock
        cnn_out = torch.cat(cnn_outs, dim=1)  # Shape: (1, num_sheet * (W/8) * (H/8))

        # Xử lý đặc trưng sản phẩm
        product_features = []
        for product in observation_products:
            length, width = product["size"]
            quantity = product["quantity"]
            # Xem xét cả hai hướng
            product_features.append([length, width, quantity])
            product_features.append([width, length, quantity])
        # Nếu số lượng sản phẩm ít hơn, thêm padding với 0
        while len(product_features) < 2 * self.max_product_type:
            product_features.append([0.0, 0.0, 0.0])
        product_features = np.array(product_features[:2 * self.max_product_type], dtype=np.float32)
        product_features = torch.from_numpy(product_features).view(1, -1).to(next(self.parameters()).device)  # Shape: (1, 6 * max_product_type)

        # Kết hợp đặc trưng từ CNN và sản phẩm
        combined_input = torch.cat([cnn_out, product_features], dim=1)  # Shape: (1, cnn_dim + product_dim)
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

        # Tính toán kích thước đầu ra của CNN
        cnn_output_dim = num_sheet * (max_w // 8 + 1) * (max_h // 8 + 1)  # Đảm bảo tính đúng kích thước
        product_output_dim = 6 * max_product_type
        input_dim = cnn_output_dim + product_output_dim

        # Các lớp fully connected
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
                    action_index = sheet * self.max_product_type + product
                    # Đánh dấu hành động hợp lệ nếu sản phẩm còn lại
                    valid_actions[action_index] = observation_products[product]["quantity"] > 0
        # Nếu không có hành động hợp lệ, đánh dấu tất cả các hành động là hợp lệ để tránh softmax nhận toàn bộ giá trị -1e9
        if not valid_actions.any():
            valid_actions = torch.ones(self.num_sheet * self.max_product_type, dtype=torch.bool).to(next(self.parameters()).device)
        return valid_actions

    def forward(self, observation_stocks, observation_products):
        combined_input = self.encoder(observation_stocks, observation_products)
        x = F.relu(self.fc1(combined_input))
        action_logits = self.fc2(x)  # Shape: (1, num_sheet * max_product_type)
        
        # Lấy mặt nạ các hành động hợp lệ
        valid_actions = self.get_valid_actions(observation_products)
        
        # Mở rộng mặt nạ để phù hợp với kích thước batch nếu cần
        if action_logits.dim() == 2:
            valid_actions = valid_actions.unsqueeze(0).expand(action_logits.size(0), -1)
        
        # Đặt logits của các hành động không hợp lệ thành một giá trị rất nhỏ
        action_logits[~valid_actions] = -1e9
        
        # Tính phân phối xác suất
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

        # Tính toán kích thước đầu ra của CNN
        cnn_output_dim = num_sheet * (max_w // 8 + 1) * (max_h // 8 + 1)  # Đảm bảo tính đúng kích thước
        product_output_dim = 6 * max_product_type
        input_dim = cnn_output_dim + product_output_dim

        # Các lớp fully connected cho giá trị
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, observation_stocks, observation_products):
        combined_input = self.encoder(observation_stocks, observation_products)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(1)  # Shape: (batch_size,)

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

    def select_action(self, observation):
        """
        Chọn một hành động dựa trên phân phối xác suất của Actor.
        """
        observation_stocks = observation["stocks"]
        observation_products = observation["products"]

        action_probs = self.actor(observation_stocks, observation_products)  # Shape: (1, num_sheet * max_product_type)
        distribution = torch.distributions.Categorical(action_probs)
        selected_action = distribution.sample()
        log_prob = distribution.log_prob(selected_action)
        entropy = distribution.entropy()

        # Lưu trữ các thông tin cần thiết cho cập nhật sau này
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
        return {
            "stock_idx": sheet_index, 
            "size": [prod_w, prod_h],
            "position": [best_pos_x, best_pos_y]
        }

    def get_action(self, observation, info): 
        """
        Chọn hành động dựa trên trạng thái hiện tại và xử lý các hành động không hợp lệ.
        """
        selected_action = self.select_action(observation)
        sheet_index = selected_action // self.max_product_type
        product_index = selected_action % self.max_product_type

        observation_stocks = observation["stocks"]
        observation_products = observation["products"]

        # Kiểm tra tính hợp lệ của product_index
        if product_index >= len(observation_products) or observation_products[product_index]["quantity"] <= 0:
            # Hành động không hợp lệ
            reward = -1.0  # Hình phạt

            # Ghi nhận phần thưởng
            self.store_reward(reward)

            return None
        else:
            # Hành động hợp lệ
            placed = self.place_product(observation, sheet_index, product_index)
            if placed is not None:
                # Đặt sản phẩm thành công
                reward = self.calculate_reward(observation)
            else:
                # Đặt sản phẩm không thành công
                reward = -1.0

            # Ghi nhận phần thưởng
            self.store_reward(reward)

            return placed

    def update(self):
        """
        Cập nhật mạng neuron dựa trên các trải nghiệm đã lưu.
        """
        if len(self.rewards) == 0:
            return  # Không có gì để cập nhật

        # Đảm bảo tất cả các danh sách đều có cùng độ dài
        assert len(self.rewards) == len(self.log_probs) == len(self.values) == len(self.entropies), \
            f"Mismatch in trajectory lengths: rewards={len(self.rewards)}, log_probs={len(self.log_probs)}, values={len(self.values)}, entropies={len(self.entropies)}"

        # Tính toán returns
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).squeeze(1).to(self.device)  # Shape: (batch_size,)

        # Chuẩn hóa returns
        returns_mean = returns.mean()
        returns_std = returns.std(unbiased=False)
        returns = (returns - returns_mean) / (returns_std + 1e-5)

        # Chuẩn bị các giá trị để tính toán lợi thế
        values = torch.stack(self.values).squeeze(1).to(self.device)      # Shape: (batch_size,)
        log_probs = torch.stack(self.log_probs).to(self.device)          # Shape: (batch_size,)
        entropies = torch.stack(self.entropies).to(self.device)          # Shape: (batch_size,)

        # Tính lợi thế (advantage)
        advantage = returns - values  # Shape: (batch_size,)

        # Tính toán mất mát cho Actor
        actor_loss = -(log_probs * advantage.detach()).mean() - 0.01 * entropies.mean()

        # Tính toán mất mát cho Critic
        critic_loss = advantage.pow(2).mean()

        # Backpropagation cho Actor
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # Backpropagation cho Critic
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # Xóa các danh sách lưu trữ sau khi cập nhật
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def store_reward(self, reward):
        """
        Lưu trữ phần thưởng để sử dụng trong quá trình cập nhật.
        """
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
        print(f"Stored reward: {reward}. Total rewards: {len(self.rewards)}")

    def calculate_reward(self, observation):
        """
        Tính phần thưởng dựa trên tỷ lệ waste_total / area_total của các sheets đã sử dụng.
        Tỷ lệ càng thấp, phần thưởng càng cao.
        """
        waste_total = 0
        area_total = 0

        for idx, sheet in enumerate(observation["stocks"]):
            # Kiểm tra xem tấm stock này đã có vật phẩm đặt hay chưa (có số >=0)
            if np.any(sheet >= 0):
                waste = np.sum(sheet == -1)  # Các ô trống
                area = np.sum(sheet != -2)
                waste_total += waste
                area_total += area
            else:
                continue

        if area_total == 0:
            waste_ratio = 0
        else:
            waste_ratio = waste_total / area_total

        # Định nghĩa hàm phần thưởng: tỷ lệ waste thấp hơn dẫn đến phần thưởng cao hơn
        reward = -waste_ratio  # Ví dụ: phần thưởng là nghịch đảo tỷ lệ waste

        # Kiểm tra để đảm bảo không có giá trị không hợp lệ
        if not np.isfinite(reward):
            print(f"Invalid reward calculated: waste_total={waste_total}, area_total={area_total}")
            reward = -1.0  # Gán một giá trị phần thưởng mặc định

        print(f"Total Waste: {waste_total}, Total Area: {area_total}, Waste Ratio: {waste_ratio}, Reward: {reward}")
        return reward

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

