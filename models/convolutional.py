# Description: Convolutional blocks for the CoatNet model
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.GELU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate, stride, shrink_rate):
        super(MBConv, self).__init__()

        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expansion_rate

        self.expand = in_channels != hidden_dim
        if self.expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn0 = nn.BatchNorm2d(hidden_dim)

        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        if shrink_rate is not None:
            self.se = SqueezeExcitation(hidden_dim, int(in_channels * shrink_rate))
        else:
            self.se = nn.Identity()
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = inputs

        if self.expand:
            x = F.gelu(self.bn0(self.expand_conv(x)))

        x = F.gelu(self.bn1(self.conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            return x + inputs
        else:
            return x
        
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=4, stride=1, shrink_rate=0.25,
                 stochastic_rate=0.2):
        super(CNN, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.mbconv = MBConv(in_channels, out_channels, expansion_rate, stride, shrink_rate)
        self.stochastic_rate = stochastic_rate
        
    def stochastic_depth(self, layer, x):
        if not self.training:
            if torch.rand(1) < self.stochastic_rate:
                return layer(x)
            else:
                return x
        else:
            return layer(x) * self.stochastic_rate
        
    def forward(self, x):
        return self.stochastic_depth(lambda m: m + self.mbconv(self.norm(m)), x)
    
class MbConv(nn.Module):
    def __init__(self, L, D, D_in, expansion_rate=4, stride=1, shrink_rate=0.25, 
                 stochastic_rate=0.2) -> None:
        super(MbConv, self).__init__()
        self.D = D
        self.L = L
        
        self.pool = nn.MaxPool2d(2, 2)
        self.proj = nn.Conv2d(D_in, D, 1)
        self.cnns = nn.Sequential(*[CNN(D, D, expansion_rate, stride, 
                                        shrink_rate, stochastic_rate) for i in range(L)])
    
    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        x = self.cnns(x)
        return x
    