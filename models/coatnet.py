
import torch.nn as nn

from .transformer import TFMRel
from .convolutional import MbConv


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Stem, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv1(self.conv0(x))
    
class CoAtNet(nn.Module):
    def __init__(self, image_size=224, num_channels=3, num_classes=1000,
                 lengths=[2, 3, 5, 2], depths=[64, 96, 192, 384, 768], 
                 sizes=[0.25, 0.125, 0.0625, 0.03125], 
                 blocks="CCTT", mbconv_e=4, mbconv_se=0.25,
                 head_dim=32, mem_eff=True, tfmrel_e=4, qkv_bias=False,
                 fc_e=2, stochastic_rate=0.2):
        super(CoAtNet, self).__init__()
        
        assert len(blocks) == len(lengths) == len(sizes), "blocks, lenghts and sizes must have the same length"
        assert len(depths) == len(sizes) + 1, "depths must have one more element than sizes"
        
        self.img_size = image_size
        
        self.S0 = Stem(num_channels, depths[0])
        self.stages = nn.Sequential()
        for L, block in enumerate(blocks):
            if block == 'C':
                self.stages.add_module(f"S{L+1}_MbConv", MbConv(L=lengths[L], D=depths[L+1], D_in=depths[L], 
                                                                expansion_rate=mbconv_e, stride=1, 
                                                                shrink_rate=mbconv_se, 
                                                                stochastic_rate=stochastic_rate))
            elif block == 'T':
                self.stages.add_module(f"S{L+1}_TFMrel", TFMRel(L=lengths[L], D=depths[L+1], D_in=depths[L], 
                                                                height=int(image_size*sizes[L]), 
                                                                width=int(image_size*sizes[L]), 
                                                                head_dim=head_dim, mem_eff=mem_eff, 
                                                                expansion_factor=tfmrel_e, qkv_bias=qkv_bias, 
                                                                stochastic_rate=stochastic_rate))
            else:
                raise ValueError(f"Block {block} not implemented in CoatNet")
        
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(
            nn.Linear(depths[-1], fc_e*depths[-1]),
            nn.GELU(),
            nn.Linear(fc_e*depths[-1], num_classes)
        )
        
    def forward(self, x):
        x = self.S0(x)
        x = self.stages(x)
        x = self.GAP(x).view(x.shape[0], -1)
        x = self.FC(x)
        return x
        
        