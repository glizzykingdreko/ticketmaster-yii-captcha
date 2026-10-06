import torch
from torch import nn

from generator import CHARSET, CODE_LENGTH

# 0.8 scale of the native 120x100
INPUT_HEIGHT = 80
INPUT_WIDTH = 96
FEATURE_WIDTH = INPUT_WIDTH // 2 ** 4

RELEASES = {
    "pico": 24,
    "nano": 32,
    "micro": 64,
}
DEFAULT_RELEASE = "micro"


def conv_block(in_channels, out_channels, count=2):
    layers = []
    for index in range(count):
        layers += [
            nn.Conv2d(
                in_channels if index == 0 else out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CaptchaNet(nn.Module):
    def __init__(self, channels=RELEASES[DEFAULT_RELEASE]):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, channels),
            conv_block(channels, channels, count=1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels * FEATURE_WIDTH, CODE_LENGTH * len(CHARSET)),
        )

    def forward(self, images):
        features = self.features(images).mean(dim=2).flatten(1)
        return self.head(features).view(-1, CODE_LENGTH, len(CHARSET))


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
