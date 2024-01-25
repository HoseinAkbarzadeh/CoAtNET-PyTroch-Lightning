import os

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageNet

from .imagenet import ImageNet1K, create_coatnet_transform


def create_train_dataloaders_old(args, val_frac, train=True):
    ds = ImageNetDS(root=os.path.join(args.data_dir, 'train' if train else 'validation'), 
                    class_dict=args.class_dict, 
                    transform=create_coatnet_transform(train=True, img_size=args.image_size))
    
    if train:
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(ds, [len(ds) - int(len(ds) * val_frac), int(len(ds) * val_frac)], 
                                        generator=gen)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    
        return train_loader, val_loader
    else:
        val_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        return val_loader
    
def create_train_dataloaders(args, val_frac, train=True):
    ds = ImageNet1K(root=args.data_dir, split='train' if train else 'val', meta_path=args.class_dict,
                  transform=create_coatnet_transform(train=train, img_size=args.image_size))
    
    if train:
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(ds, [len(ds) - int(len(ds) * val_frac), int(len(ds) * val_frac)], 
                                        generator=gen)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    
        return train_loader, val_loader, len(train_ds)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        return loader, len(ds)