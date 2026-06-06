import os
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.dataset import get_dataloaders
from training.lightning_module import MambaPedOSAModule

def train_model(config_path, data_dir, gpus, fast_dev_run, run_name, **kwargs):
    print(f"\n{'='*50}\nStarting Training: {run_name}\n{'='*50}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    train_loader, val_loader, _, _ = get_dataloaders(data_dir)
    
    model = MambaPedOSAModule(config)
    
    # WandB Logger
    logger = WandbLogger(
        project='Mamba-PedOSA',
        name=run_name,
        offline=True # Use offline by default to avoid hanging if user has no wandb account configured
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'models/checkpoints/{run_name}',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_auprc_mild',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_auprc_mild',
        patience=20,
        verbose=True,
        mode='max'
    )
    
    trainer = pl.Trainer(
        max_epochs=kwargs.get('epochs', 150),
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        fast_dev_run=fast_dev_run
    )
    
    trainer.fit(model, train_loader, val_loader)
    print(f"Finished {run_name}. Best checkpoint saved at: {checkpoint_callback.best_model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Path to main model config')
    parser.add_argument('--ablation', action='store_true', help='Run ablation experiments automatically')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--epochs', type=int, default=150, help='Max epochs')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a quick PyTorch Lightning dev run')
    parser.add_argument('--data_dir', type=str, default='data/features', help='Path to dataset directory')
    args = parser.parse_args()
    
    # 1. Main Training
    train_model(args.config, args.data_dir, args.gpus, args.fast_dev_run, run_name='main_mamba_pedosa', epochs=args.epochs)
    
    # 2. Auto-run Ablations
    if args.ablation:
        ablation_configs = [
            'configs/ablations/ablation_ecg_only.yaml',
            'configs/ablations/ablation_ecg_ptt.yaml',
            'configs/ablations/ablation_no_jepa.yaml',
            'configs/ablations/ablation_unidirectional.yaml',
            'configs/ablations/ablation_mse_head.yaml'
        ]
        
        for abl_cfg in ablation_configs:
            if os.path.exists(abl_cfg):
                run_name = os.path.splitext(os.path.basename(abl_cfg))[0]
                train_model(abl_cfg, args.data_dir, args.gpus, args.fast_dev_run, run_name=run_name, epochs=args.epochs)
            else:
                print(f"Warning: Ablation config {abl_cfg} not found. Skipping.")

if __name__ == '__main__':
    main()
