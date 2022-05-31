import Consts
import torch.nn as nn

num_features = Consts.num_features
comp_length = Consts.comp_length
seed_size = Consts.seed_size

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 128

        # fc1 - transform the seed into an initial song shape
        self.fc1 = nn.Sequential(
            nn.Linear(seed_size, 2 * seed_size, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * seed_size, 3 * seed_size, bias=False)
        )

        self.bn1 = nn.BatchNorm1d(num_features)

        # convolution 1
        self.conv1_kernel_size = 32
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_features * 2,
                               kernel_size=self.conv1_kernel_size, padding="same", padding_mode="zeros")

        # fc 2
        self.fc2_in_size = comp_length * num_features
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc2_in_size, self.fc2_in_size, bias=False),
            nn.LeakyReLU(0.2)
        )

        # convolution 2
        self.conv2_comp_length = comp_length
        self.conv2_kernel_size = 24
        self.conv2 = nn.Conv1d(in_channels=num_features * 2, out_channels=num_features * 3,
                               kernel_size=self.conv2_kernel_size, padding="same", padding_mode="zeros")

        # fc 3
        self.fc3_in_size = (
                               self.conv2_comp_length) * num_features * 3  # each convolution loses (kernel_size) features (because of the layer edges)
        self.fc3 = nn.Sequential(
            nn.LeakyReLU(0.2)
        )

        # convolution 3
        self.conv3_comp_length = self.conv2_comp_length
        self.conv3_kernel_size = 20
        self.conv3 = nn.Conv1d(in_channels=num_features * 3, out_channels=num_features * 4,
                               kernel_size=self.conv3_kernel_size, padding="same", padding_mode="zeros")

        # fc 4
        self.fc4_in_size = self.conv3_comp_length * num_features * 4
        self.fc4 = nn.Sequential(
            nn.LeakyReLU(0.2)
        )

        # convolution 4
        self.conv4_comp_length = self.conv3_comp_length
        self.conv4_kernel_size = 8
        self.conv4 = nn.Conv1d(in_channels=num_features * 4, out_channels=num_features * num_features,
                               kernel_size=self.conv4_kernel_size, padding="same", padding_mode="zeros")
        self.mp4 = nn.MaxPool1d(num_features)

        # fc 5
        self.fc5_in_size = comp_length * num_features
        self.fc5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(self.fc5_in_size, comp_length * num_features, bias=False),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(-1, num_features, comp_length)
        x = self.bn1(x)
        x = self.conv1(x)
        x = x.reshape(-1, self.fc2_in_size)
        x = self.fc2(x)
        x = x.reshape(-1, num_features * 2, self.conv2_comp_length)
        x = self.conv2(x)
        x = x.reshape(-1, self.fc3_in_size)
        x = self.fc3(x)
        x = x.reshape(-1, num_features * 3, self.conv3_comp_length)
        x = self.conv3(x)
        x = x.reshape(-1, self.fc4_in_size)
        x = self.fc4(x)
        x = x.reshape(-1, num_features * 4, self.conv4_comp_length)
        x = self.mp4(self.conv4(x))
        x = x.reshape(-1, self.fc5_in_size)
        x = self.fc5(x)
        return x