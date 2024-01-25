import os
from argparse import ArgumentParser

import torch

from dataset import create_train_dataloaders
from utils import DotDict
from models.coatnet import CoAtNet
slurm_state = True if 'SLURM_JOB_ID' in os.environ else False

if slurm_state:
    version = 'slurm'+os.environ['SLURM_JOB_ID']
else:
    from datetime import datetime
    version = datetime.now().strftime('%d-%b-%y_%H-%M')

def parse_cmd():
    parser = ArgumentParser('PyTorch CoatNET training script.')
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing ImageNet dataset.')
    parser.add_argument('--class_dict', type=str, required=True, 
                        help='Path to the pickle file containing the classes mapping dictionary.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config toml file.')
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                        help='Directory to save checkpoints.')
    parser.add_argument('--log_dir', type=str, required=True,
                        help="The tensorboard logging directory.")
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Base learning rate.')
    parser.add_argument('--learning_schedule', type=str, default='cosine',
                        help='Learning rate schedule.')
    parser.add_argument('--ema_decay', type=float, default=None,
                        help='Exponential moving average decay.')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='Weight decay.')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing.')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Mixup alpha.')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs to train.')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use.')
    
    return parser.parse_args()

def train(args):
    
    # prepare dataset loaders
    # train_loader, val_loader = create_train_dataloaders(args, val_frac=0.05, train=True)
    train_loader = create_train_dataloaders(args, val_frac=0.05, train=False)
        
    # create the model and lightning wrapper
    model = CoAtNet(image_size=args.MODEL.image_size, num_channels=args.MODEL.num_channels,
                             num_classes=args.MODEL.num_classes, lengths=args.MODEL.lengths, 
                             depths=args.MODEL.depths, sizes=args.MODEL.sizes, blocks=args.MODEL.blocks, 
                             mbconv_e=args.MODEL.mbconv_e, mbconv_se=args.MODEL.mbconv_se, 
                             head_dim=args.MODEL.head_dim, mem_eff=args.MODEL.mem_eff, 
                             tfmrel_e=args.MODEL.tfmrel_e, qkv_bias=args.MODEL.qkv_bias, 
                             fc_e=args.MODEL.fc_e, stochastic_rate=args.MODEL.stochastic_rate).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
        
    log_interval = 100
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = []
        for itr, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if itr % log_interval == 0:
                print(f"Epoch {epoch+1} iteration {itr} train loss: {loss.item()}")
        print(f"Epoch {epoch+1} train loss: {sum(train_loss)/len(train_loss)}")
        
        # model.eval()
        # val_loss = []
        # for X, y in val_loader:
        #     X, y = X.cuda(), y.cuda()
        #     y_hat = model(X)
        #     loss = criterion(y_hat, y)
        #     val_loss.append(loss.item())
        # print(f"Epoch {epoch+1} val loss: {sum(val_loss)/len(val_loss)}")

if __name__ == '__main__':
    conf = parse_cmd()
    args = DotDict.from_toml(conf.config)
    args.update(**conf.__dict__)
    
    train(args)
    
    
    
    
    