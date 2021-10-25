import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(path) -> nn.Module:
    model = BinaryClassifier()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.max_p_1 = F.max_pool2d
        self.activation_1 = F.relu
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.max_p_2 = F.max_pool2d
        self.activation_2 = F.relu
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7680, 1024)
        self.fc2 = nn.Linear(1024, 37)

    def forward(self, x):
        batch = x.shape[0]
        x = self.conv1_drop(self.activation_1( self.max_p_1 ( self.conv1_bn ( self.conv1(x)), 2)))
        x = self.conv2_drop(self.activation_2( self.max_p_2 ( self.conv2_bn ( self.conv2(x)), 2)))
        x = x.reshape(batch, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
