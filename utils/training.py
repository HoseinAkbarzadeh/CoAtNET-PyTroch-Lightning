import math
from torch.optim.lr_scheduler import _LRScheduler
        
class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, Tmax, eta_min, warmup_start_lr, max_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.Tmax = Tmax
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = [self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_steps for _ in self.base_lrs]
        else:
            lr = [self.eta_min + ( self.max_lr - self.eta_min ) * (1 + math.cos(math.pi * ( self.last_epoch - self.warmup_steps) / ( self.Tmax - self.warmup_steps))) / 2 for _ in self.base_lrs]
        return lr
    
class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, Tmax, end_factor, warmup_start_lr, max_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.Tmax = Tmax
        self.end_factor = end_factor
        self.warmup_start_lr = warmup_start_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = [self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_steps for _ in self.base_lrs]
        else:
            lr = [self.max_lr + (self.end_factor - self.max_lr) * ( self.last_epoch - self.warmup_steps) / ( self.Tmax - self.warmup_steps) for _ in self.base_lrs]
        return lr