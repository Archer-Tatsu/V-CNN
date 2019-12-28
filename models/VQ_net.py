import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
from torchvision.models import densenet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.leaky = 0.1

        self.group_layers = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True)
        )
        densenet_layers = densenet.DenseNet(num_init_features=96, growth_rate=16, block_config=(4, 8, 16),
                                            drop_rate=0.5)
        densenet_layers.features[0] = nn.Conv2d(64, 96, kernel_size=7, stride=2, padding=3, bias=False)

        self.shared_layers = nn.Sequential(
            densenet_layers.features,
            nn.Conv2d(360, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky, inplace=True),
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        self.score_layers = nn.Sequential(
            nn.Conv2d(448, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(self.leaky, inplace=True),
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.em_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky, inplace=True),
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 2, stride=2, padding=1, bias=False),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
        :param x: Impaired viewports. shape: (batch_size, channels, height, width)
        :param y: Viewport error map with the same shape of x.
        """

        x = torch.cat((x, y), dim=1)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        batch_size = x.shape[0]
        x = self.group_layers(x)
        x = self.shared_layers(x)

        em = self.em_layers(x)

        size = em.size()
        em = em.view(size[0], size[1], -1)
        em = self.softmax(em)
        em = em.view(size)

        z = F.interpolate(y, x.shape[-2:], mode='bilinear', align_corners=False)
        z = z.repeat(1, 64, 1, 1)
        x = torch.cat((x, z), dim=1)
        x = self.score_layers(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x, em
