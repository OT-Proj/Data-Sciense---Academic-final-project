import Consts
import torch.nn as nn

num_features = Consts.num_features
comp_length = Consts.comp_length


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_size = comp_length * num_features
        self.layer_multiplier = 128

        self.bn1 = nn.BatchNorm1d(num_features)

        # convolution 1
        self.conv1_kernel_size = 32
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_features * 2,
                               kernel_size=self.conv1_kernel_size, padding="same", padding_mode="zeros")
        self.mp1 = nn.MaxPool1d(num_features)

        # fc1
        self.fc1_in_size = comp_length * num_features * 2
        self.fc1 = nn.Sequential(
            nn.Linear(self.fc1_in_size, self.fc1_in_size),
            nn.LeakyReLU(0.2),
        )

        # convolution 2
        self.conv2_kernel_size = 32
        self.conv2 = nn.Conv1d(in_channels=num_features * 2, out_channels=num_features * 4,
                               kernel_size=self.conv2_kernel_size, padding="same", padding_mode="zeros")
        self.mp2 = nn.MaxPool1d(num_features)

        # fc2
        self.fc2_in_size = (comp_length) * num_features * 4
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc2_in_size, self.fc2_in_size),
            nn.LeakyReLU(0.2)
        )

        # convolution 3
        self.conv3_kernel_size = 8
        self.conv3 = nn.Conv1d(in_channels=num_features * 4, out_channels=num_features * num_features,
                               kernel_size=self.conv3_kernel_size, padding="same", padding_mode="zeros")
        self.mp3 = nn.MaxPool1d(num_features)

        # fc3
        self.fc3_in_size = comp_length * num_features
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc3_in_size, self.fc3_in_size),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.fc3_in_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1, num_features, comp_length)
        x = self.conv1(x)
        x = x.reshape(-1, self.fc1_in_size)
        x = self.fc1(x)
        x = x.reshape(-1, num_features * 2, comp_length)
        x = self.conv2(x)
        x = x.reshape(-1, self.fc2_in_size)
        x = self.fc2(x)
        x = x.reshape(-1, num_features * 4, comp_length)
        x = self.mp3(self.conv3(x))
        x = x.reshape(-1, self.fc3_in_size)
        x = self.fc3(x)
        return x
