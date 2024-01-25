from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
from torch.distributions.beta import Beta
import copy

from models.coatnet import CoAtNet
from utils.training import WarmupCosineLR, WarmupLinearLR

class LightningCoatNet(LightningModule):
    def __init__(self, image_size, num_channels, num_classes, lengths, depths, 
                 sizes, blocks, mbconv_e, mbconv_se, head_dim, mem_eff, tfmrel_e, 
                 qkv_bias, fc_e, stochastic_rate, learning_rate=5e-5, learning_schedule=None,
                 ema_decay=None, weight_decay=1e-8, label_smoothing=0.1, alpha=0.8,
                 peak_lr=None, warmup_steps=None, min_lr=None, max_steps=None):
        super(LightningCoatNet, self).__init__() 
        self.net = CoAtNet(image_size=image_size, num_channels=num_channels, num_classes=num_classes,
                           lengths=lengths, depths=depths, sizes=sizes, blocks=blocks, mbconv_e=mbconv_e,
                           mbconv_se=mbconv_se, head_dim=head_dim, mem_eff=mem_eff,
                           tfmrel_e=tfmrel_e, qkv_bias=qkv_bias, fc_e=fc_e, stochastic_rate=stochastic_rate)
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.ema_decay = ema_decay
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        
        if self.learning_schedule is not None:
            if peak_lr is None or warmup_steps is None or min_lr is None or max_steps is None:
                raise ValueError("peak_lr, warmup_steps, max_steps and min_lr must be specified for learning schedule")
        
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        if self.ema_decay is not None:
            self.ema_model = copy.deepcopy(self.net)  # EMA model
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y_a, y_b, lam = self.mixup_data(x, y, self.alpha)
        y_hat = self(x)
        loss = self.mixup_criterion(y_hat, y_a, y_b, lam)

        if self.ema_decay is not None:
            self.update_ema_weights()
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.ema_decay is not None:
            y_hat = self.ema_model(x)  # Use EMA model for validation
        else:
            y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.learning_rate, 
                                      weight_decay=self.weight_decay)
        if self.learning_schedule is None:
            return optimizer
            
        if self.learning_schedule == 'cosine':
            lr_scheduler = WarmupCosineLR(optimizer, 
                                          warmup_steps=self.warmup_steps, 
                                          Tmax=self.max_steps, 
                                          eta_min=self.peak_lr, 
                                          warmup_start_lr=self.min_lr, 
                                          max_lr=self.peak_lr)
        elif self.learning_schedule == 'linear':
            lr_scheduler = WarmupLinearLR(optimizer, 
                                          warmup_steps=self.warmup_steps, 
                                          Tmax=self.max_steps, 
                                          end_factor=self.min_lr, 
                                          warmup_start_lr=self.peak_lr, 
                                          max_lr=self.min_lr)
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': lr_scheduler, 
                'interval': 'step',
                'frequency': 1,
            }
        }

    def update_ema_weights(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.net.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    @staticmethod
    def mixup_data(imgs, labels, alpha=0.8):
        batch_size = imgs.size(0)
        device = imgs.device  # Get the device from imgs

        lam = Beta(torch.tensor(alpha).to(device), torch.tensor(alpha).to(device)).sample().item()
        index = torch.randperm(batch_size).to(device)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[index, :]
        label_a, label_b = labels, labels[index]
        return mixed_imgs, label_a, label_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)