import torch
import torch.nn as nn

class hw1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(hw1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feature = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.feature(x)

#减少模型参数
class hw1_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(hw1_2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feature = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),

            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.feature(x)

class hw1_3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(hw1_3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feature = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.feature(x)