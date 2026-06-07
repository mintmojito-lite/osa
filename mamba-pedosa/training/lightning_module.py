import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from models.mamba_pedosa import MambaPedOSA
from training.loss import PedOSALoss
from training.metrics import compute_auprc, compute_brier_scores, compute_ece, compute_mcc

class MambaPedOSAModule(pl.LightningModule):
    def __init__(self, config=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MambaPedOSA(config)
        self.lr = lr
        
        # severity classification weights (0=Normal, 1=Mild, 2=Moderate, 3=Severe)
        # We don't need focal_weights for the current EvidentialLoss implementation
        
        self.criterion = PedOSALoss(evidential_coeff=1.0, ahi_coeff=0.5, sw_coeff=0.3)
        
        self.val_step_outputs = []

    def forward(self, ecg, spo2, ptt, clinical):
        return self.model(ecg, spo2, ptt, clinical)

    def training_step(self, batch, batch_idx):
        ecg = batch['ecg']
        spo2 = batch['spo2']
        ptt = batch['ptt_swing']
        clinical = batch['clinical']
        
        targets = {
            'ahi_label': batch['ahi_label'],
            'severity_class': batch['severity_class']
        }
        
        preds = self(ecg, spo2, ptt, clinical)
        loss_dict = self.criterion(preds, targets)
        
        loss = loss_dict['total_loss']
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_evidential_loss', loss_dict['evidential_loss'], on_step=False, on_epoch=True)
        self.log('train_cls_loss', loss_dict['cls_loss'], on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        ecg = batch['ecg']
        spo2 = batch['spo2']
        ptt = batch['ptt_swing']
        clinical = batch['clinical']
        
        targets = {
            'ahi_label': batch['ahi_label'],
            'severity_class': batch['severity_class']
        }
        
        preds = self(ecg, spo2, ptt, clinical)
        loss_dict = self.criterion(preds, targets)
        
        self.log('val_loss', loss_dict['total_loss'], prog_bar=True, on_epoch=True)
        
        # Store for epoch-level metrics
        self.val_step_outputs.append({
            'ahi_pred': preds['ahi_pred'].detach().cpu(),
            'probs': preds['prob'].detach().cpu(),
            'targets_ahi': targets['ahi_label'].detach().cpu(),
            'targets_cls': targets['severity_class'].detach().cpu()
        })
        
        return loss_dict['total_loss']

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
            
        ahi_pred = torch.cat([x['ahi_pred'] for x in self.val_step_outputs]).numpy()
        probs = torch.cat([x['probs'] for x in self.val_step_outputs]).numpy()
        targets_ahi = torch.cat([x['targets_ahi'] for x in self.val_step_outputs]).numpy()
        targets_cls = torch.cat([x['targets_cls'] for x in self.val_step_outputs]).numpy()
        
        # 1. Continuous AHI MAE
        mae = np.mean(np.abs(ahi_pred - targets_ahi))
        self.log('val_mae', mae, prog_bar=True)
        
        # 2. Brier Score
        brier_scores = compute_brier_scores(probs, targets_cls)
        for k, v in brier_scores.items():
            self.log(f'val_{k}', v)
            
        # 3. Expected Calibration Error
        ece = compute_ece(probs, targets_cls)
        self.log('val_ece', ece, prog_bar=True)
        
        # 4. MCC
        preds_cls = np.argmax(probs, axis=1)
        mcc = compute_mcc(preds_cls, targets_cls)
        self.log('val_mcc', mcc)
        
        # 5. AUPRC
        auprc_scores = compute_auprc(probs, targets_cls)
        for k, v in auprc_scores.items():
            self.log(f'val_{k}', v)
            if k == 'auprc_class_1':
                # Mild vs rest is the primary tracked metric for early stopping
                self.log('val_auprc_mild', v, prog_bar=True)
                
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
        
    def on_train_epoch_end(self):
        # Unfreeze encoder with lr/10 logic after epoch 10
        if self.current_epoch == 10:
            print(f"Epoch {self.current_epoch}: Unfreezing encoder layers...")
            # We assume the user wants the entire encoder unfrozen
            for param in self.model.encoder.parameters():
                param.requires_grad = True
                
            # Update optimizer with new parameter group for encoder
            # Lightning handles optimizers, so we modify the current optimizer's param groups
            # First, filter out encoder params from the base group
            base_params = [p for p in self.model.parameters() if id(p) not in [id(ep) for ep in self.model.encoder.parameters()]]
            encoder_params = list(self.model.encoder.parameters())
            
            # Since lightning wraps the optimizer, this is a bit hacky but works for param grouping
            opt = self.optimizers()
            opt.param_groups[0]['params'] = base_params
            opt.add_param_group({'params': encoder_params, 'lr': self.lr / 10.0})
