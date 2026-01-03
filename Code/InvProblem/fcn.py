import torch.nn as nn
import torch.nn.functional as F
from resblock import ResBlock
from cbam import CBAM

class FCN(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            ResBlock(32),
            ResBlock(32),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResBlock(64),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.cbam = CBAM(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.regressor(x)
