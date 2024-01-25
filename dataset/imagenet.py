import os

import torch
from torch.distributions.beta import Beta
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, CenterCrop, RandomResizedCrop, Normalize, ToTensor
from timm.data import rand_augment_transform


IMG_EXTENSIONS = (".jpg", ".jpeg")

def create_coatnet_transform(train=True, img_size=224, mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225], randaug_m=15, rand_aug_n=2):
    return Compose([
        CenterCrop(img_size) if train else RandomResizedCrop(img_size),
        rand_augment_transform(f"rand-m{randaug_m}-n{rand_aug_n}"),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    
def mix_up_augment(imgs, labels, alpha=0.8):
    batch_size = imgs.size(0)
    lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    index = torch.randperm(batch_size)
    imgs = lam * imgs + (1 - lam) * imgs[index, :]
    label_a, label_b = labels, labels[index]
    return imgs, label_a, label_b, lam

    
class ImageNet1K(ImageFolder):
    def __init__(self, root, meta_path=None, split='train', **kwargs):
        self.split = 'train' if split == 'train' else 'validation'
        super().__init__(os.path.join(root, self.split), **kwargs)
        
        if meta_path is None:
            meta_path = 'dataset/meta.bin'
        
        wnid_to_classes = torch.load(meta_path, weights_only=True)[0]
        
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}
