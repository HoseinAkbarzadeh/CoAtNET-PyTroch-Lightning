import os
from argparse import ArgumentParser

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import create_train_dataloaders
from utils import DotDict
from utils.lightning import LightningCoatNet

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


def resume_train(args):
    
    # prepare dataset loaders
    train_loader, val_loader = create_train_dataloaders(args, val_frac=0.05, train=True)
    
    # checkpoint callback
    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=args.MODEL.name+'-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        save_last=True,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )
    
    resume_checkpoint_path = os.path.join(args.ckpt_dir, "last.ckpt")

    
    # tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.MODEL.name,
        version=version,
        default_hp_metric=False
    )
    
    # learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # create the model and lightning wrapper
    model = LightningCoatNet.load_from_checkpoint(resume_checkpoint_path)
    
    trainer = Trainer(
        devices=args.gpus,
        num_nodes=1,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        # precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[ckpt_callback, lr_monitor, RichProgressBar()],
        logger=tb_logger,
        # progress_bar_refresh_rate=1 if slurm_state else 20,
        strategy='ddp',
        # num_sanity_val_steps=0 if slurm_state else 2,
        # sync_batchnorm=True if args.gpus > 1 else False,
        # resume_from_checkpoint=args.resume_from_checkpoint,
        # deterministic=True,
        # benchmark=True,
        # profiler='simple',
        # plugins='ddp_sharded'
        gradient_clip_algorithm='value',
        gradient_clip_val=1.0,
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    conf = parse_cmd()
    args = DotDict.from_toml(conf.config)
    args.update(**conf.__dict__)
    
    resume_train(args)
    
    
    
    
    