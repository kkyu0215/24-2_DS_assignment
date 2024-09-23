import torch
from torch import nn
from torch.nn import functional as F


class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size = 5):
        super(ConvMixerBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size = kernel_size, groups = dim, padding = 2)
        self.gelu = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(dim)


    def forward(self, x):

        residual = x

        x = self.depthwise_conv(x)
        x = self.gelu(x)
        x = self.batch_norm(x)

        x += residual

        return x


class ConvMixer(nn.Module):
    def __init__(self, input_channels = 1, num_classes = 3, dim = 64, depth = 8, patch_size = 2, kernel_size = 5):
        super(ConvMixer, self).__init__()

        self.patch_embed = nn.Conv2d(input_channels, dim, kernel_size = patch_size, stride = patch_size)
        self.conv_blocks = nn.Sequential(*[ConvMixerBlock(dim = dim, kernel_size = kernel_size) for _ in range(depth)])
        
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size = 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, num_classes)   # single value per channel
    

    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)
        
        x = self.patch_embed(x)
        x = self.conv_blocks(x)
        x = self.pointwise_conv(x)
        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
