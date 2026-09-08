import numpy as np
import torch
import torch.nn as nn

SEED = 42

class DQNPositionalEncoding(torch.nn.Module):
    def __init__(self, env, learning_rate=1e-3, device='cpu', dataset='glio'):
        super(DQNPositionalEncoding, self).__init__()
        self.device = device
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        
        # UPDATED: Now has 3 input channels (image + 2 position channels)
        if dataset == 'polyp':
            input_channels = 5
        else:
            input_channels = 3  # Changed from 1 to 3
        height, width = env.observation_space.shape[1], env.observation_space.shape[2]
        
        ### Construction of the neural network
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            n_flatten = self.features(dummy_input).view(1, -1).size(1)
            
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs)
        )
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
        if self.device == 'cuda' or self.device == 'mps':
            self.to(self.device)
            
    def forward(self, x):
        # x shape: (batch_size, 3, 60, 60)  # Updated comment
        features = self.features(x)
        features_flat = features.view(x.size(0), -1)
        q_values = self.fc(features_flat)
        return q_values
    
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            if qvals.dim() == 2 and qvals.size(0) == 1:
                action = torch.argmax(qvals, dim=-1).item()
            else:
                action = torch.argmax(qvals, dim=-1)[0].item()

        return int(action)
    
    def get_qvals(self, state):
        # 1. Handle Numpy Inputs
        if isinstance(state, np.ndarray):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            # Handle Single observation (3, 60, 60) -> (1, 3, 60, 60)
            if state_t.ndim == 3:
                state_t = state_t.unsqueeze(0)
                
        # 2. Handle Tensor Inputs (This is what comes from calculate_loss)
        elif isinstance(state, torch.Tensor):
            state_t = state.to(self.device)
            if state_t.ndim == 3:
                state_t = state_t.unsqueeze(0)
        else:
            # Fallback for lists, etc.
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        # 3. Forward pass
        qvals = self.forward(state_t)
        return qvals
    