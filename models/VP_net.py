import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2cnn import s2_near_identity_grid, S2Convolution, SO3Convolution, \
    so3_near_identity_grid


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.leaky_alpha = 0.1

        # S2 layer
        grid = s2_near_identity_grid(max_beta=np.pi / 64, n_alpha=4, n_beta=2)
        self.layer0 = nn.Sequential(
            S2Convolution(3, 16, 128, 64, grid),
            nn.GroupNorm(1, 16),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
        )

        self.flow_layer0 = nn.Sequential(
            S2Convolution(2, 16, 128, 64, grid),
            nn.GroupNorm(1, 16),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
        )

        grid = so3_near_identity_grid(max_beta=np.pi / 32, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer1, self.flow_layer1 = (
            nn.Sequential(
                SO3Convolution(16, 16, 64, 32, grid),
                nn.GroupNorm(1, 16),
                nn.LeakyReLU(self.leaky_alpha, inplace=True),
                SO3Convolution(16, 32, 32, 32, grid),
                nn.GroupNorm(2, 32),
                nn.LeakyReLU(self.leaky_alpha, inplace=True),
            )
            for _ in range(2)
        )

        grid = so3_near_identity_grid(max_beta=np.pi / 16, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer2, self.flow_layer2 = (
            nn.Sequential(
                SO3Convolution(32, 32, 32, 16, grid),
                nn.GroupNorm(2, 32),
                nn.LeakyReLU(self.leaky_alpha, inplace=True),
                SO3Convolution(32, 64, 16, 16, grid),
                nn.GroupNorm(4, 64),
                nn.LeakyReLU(self.leaky_alpha, inplace=True),
            )
            for _ in range(2)
        )

        grid = so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer3, self.flow_layer3 = (
            nn.Sequential(
                SO3Convolution(64, 64, 16, 8, grid),
                nn.GroupNorm(4, 64),
                nn.LeakyReLU(self.leaky_alpha, inplace=True),
                SO3Convolution(64, 128, 8, 8, grid),
                nn.GroupNorm(8, 128),
                nn.LeakyReLU(self.leaky_alpha, inplace=True),
            )
            for _ in range(2)
        )

        grid = so3_near_identity_grid(max_beta=np.pi / 16, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer4 = nn.Sequential(
            SO3Convolution(256, 128, 8, 8, grid),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
        )

        self.weight_layer = nn.Sequential(
            nn.Conv2d(129, 1, kernel_size=1, stride=1, bias=False),
        )

        self.refine_layer = nn.Sequential(
            nn.Conv2d(129, 2, kernel_size=1, stride=1, bias=False),
        )

        self.motion_layer1 = nn.Sequential(
            nn.Conv2d(256, 32, 3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
            nn.Conv2d(32, 8, 3, stride=2, padding=1, bias=False),
        )
        self.motion_layer2 = nn.Linear(128, 2, bias=False)
        self.control_layer = nn.Sequential(
            nn.Conv2d(128, 129, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, flow, cb):
        batch_size = img.shape[0]

        for layer in (self.layer0, self.layer1, self.layer2, self.layer3):
            img = layer(img)

        for layer in (self.flow_layer0, self.flow_layer1, self.flow_layer2, self.flow_layer3):
            flow = layer(flow)

        spatial_feat = self.layer4(torch.cat((img, flow), dim=1))
        spatial_feat = spatial_feat.mean(-1)

        motion = torch.cat((spatial_feat, flow.mean(-1)), dim=1)

        motion = self.motion_layer1(motion)
        motion = motion.reshape(batch_size, -1)

        m_control = motion.detach().unsqueeze(-1).unsqueeze(-1)
        m_control = self.control_layer(m_control)

        cb = F.adaptive_avg_pool2d(cb, spatial_feat.shape[-2:])
        spatial_feat = torch.cat((spatial_feat, cb), dim=1)
        spatial_feat = spatial_feat * m_control

        motion = self.motion_layer2(motion)

        # HM refinement.
        pred_offset = self.refine_layer(spatial_feat)
        pred_offset = pred_offset.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        motion = F.softmax(motion, dim=1)

        pred_weight = self.weight_layer(spatial_feat)

        size = pred_weight.size()
        pred_weight = pred_weight.view(size[0], size[1], -1)
        pred_weight = self.softmax(pred_weight)
        pred_weight = pred_weight.view(size)

        return pred_weight, pred_offset, motion
