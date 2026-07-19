import numpy as np
import torch
import torch.nn as nn

from .srm_filter_kernel import all_normalized_hpf_list


class MLP(nn.Module):
    def __init__(self, in_channels=64, num_classes=1):
        super().__init__()
        hidden = in_channels // 2
        self.conv_blocks0 = self._make_conv_block(in_channels, hidden, 4)
        self.conv_blocks1 = self._make_conv_block(hidden, hidden, 2)
        self.conv_blocks2 = self._make_conv_block(hidden, hidden, 2)
        self.conv_blocks3 = self._make_conv_block(hidden, hidden, 2)
        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden, num_classes)

    @staticmethod
    def _make_conv_block(in_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_blocks0(x)
        x = self.pool(x)
        x = self.conv_blocks1(x)
        x = self.pool(x)
        x = self.conv_blocks2(x)
        x = self.pool(x)
        x = self.conv_blocks3(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)


class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        filters = []
        for hpf in all_normalized_hpf_list:
            if hpf.shape[0] == 3:
                hpf = np.pad(hpf, pad_width=((1, 1), (1, 1)), mode="constant")
            filters.append(hpf)
        weight = torch.tensor(np.array(filters), dtype=torch.float32).view(30, 1, 5, 5)
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return self.hpf(x)


class LearnableFrequencyAttention(nn.Module):
    def __init__(self, channels=32, groups=8, size=256):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups
        self.channels = channels
        self.group_channels = channels // groups
        self.high_freq_mask = nn.Parameter(torch.zeros(groups, size, size))
        self.low_freq_mask = nn.Parameter(torch.zeros(groups, size, size))
        self.high_freq_weights = nn.Parameter(torch.zeros(groups, self.group_channels))
        self.low_freq_weights = nn.Parameter(torch.zeros(groups, self.group_channels))
        self._init_masks(size)
        nn.init.xavier_uniform_(self.high_freq_weights)
        nn.init.xavier_uniform_(self.low_freq_weights)

    def _init_masks(self, size):
        y = torch.linspace(-1, 1, size).view(-1, 1)
        x = torch.linspace(-1, 1, size).view(1, -1)
        distance = torch.sqrt(y**2 + x**2)
        distance = distance / distance.max()
        self.high_freq_mask.data = (5.0 * (1 - distance)).unsqueeze(0).repeat(self.groups, 1, 1)
        self.low_freq_mask.data = (5.0 * distance).unsqueeze(0).repeat(self.groups, 1, 1)

    def forward(self, rich_feat, poor_feat):
        batch, channels, height, width = rich_feat.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")

        rich_fft = torch.fft.fftshift(torch.fft.fft2(rich_feat, norm="ortho"), dim=(-2, -1))
        poor_fft = torch.fft.fftshift(torch.fft.fft2(poor_feat, norm="ortho"), dim=(-2, -1))
        rich_fft = rich_fft.view(batch, self.groups, self.group_channels, height, width)
        poor_fft = poor_fft.view(batch, self.groups, self.group_channels, height, width)

        high_mask = torch.sigmoid(self.high_freq_mask).unsqueeze(0).unsqueeze(2)
        low_mask = torch.sigmoid(self.low_freq_mask).unsqueeze(0).unsqueeze(2)
        rich_high_energy = (rich_fft * high_mask).abs().sum(dim=(-1, -2))
        poor_low_energy = (poor_fft * low_mask).abs().sum(dim=(-1, -2))

        high_att = torch.sigmoid(self.high_freq_weights.unsqueeze(0) * rich_high_energy)
        low_att = torch.sigmoid(self.low_freq_weights.unsqueeze(0) * poor_low_energy)

        rich_feat = rich_feat.view(batch, self.groups, self.group_channels, height, width)
        poor_feat = poor_feat.view(batch, self.groups, self.group_channels, height, width)
        rich_feat = (rich_feat * high_att.unsqueeze(-1).unsqueeze(-1)).view(batch, channels, height, width)
        poor_feat = (poor_feat * low_att.unsqueeze(-1).unsqueeze(-1)).view(batch, channels, height, width)
        return rich_feat, poor_feat


class AiDetFFT(nn.Module):
    """AiDet_FFT binary classifier.

    Input shape: ``(B, 2, 3, 256, 256)``. The two views are the poor-texture
    and rich-texture reconstructed images produced by ``preprocess_image``.
    """

    def __init__(self):
        super().__init__()
        self.hpf = SRM()
        self.rich_fexture = nn.Sequential(
            nn.Conv2d(90, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.poor_fexture = nn.Sequential(
            nn.Conv2d(90, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.freAttent = LearnableFrequencyAttention(groups=8)
        self.fc = MLP()

    def forward(self, x):
        poor_x = x[:, 0]
        rich_x = x[:, 1]
        batch, channels, height, width = rich_x.shape

        rich_x = rich_x.reshape(-1, 1, height, width)
        poor_x = poor_x.reshape(-1, 1, height, width)
        rich_x = self.hpf(rich_x).reshape(batch, channels * 30, height, width)
        poor_x = self.hpf(poor_x).reshape(batch, channels * 30, height, width)

        rich_feat = self.rich_fexture(rich_x)
        poor_feat = self.poor_fexture(poor_x)
        rich_feat, poor_feat = self.freAttent(rich_feat, poor_feat)
        return self.fc(torch.cat([rich_feat, poor_feat], dim=1))
