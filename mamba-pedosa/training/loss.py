import torch
import torch.nn as nn

class EvidentialClassificationLoss(nn.Module):
    """
    Negative Log-Likelihood of Dirichlet distribution
    with KL divergence regularization to penalize overconfidence on incorrect predictions.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, alpha, y_onehot, lam=0.1):
        # y_onehot is the target severity class in one-hot format
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Log-likelihood loss
        nll = torch.sum(y_onehot * (torch.log(S) - torch.log(alpha)), dim=1)
        
        # KL Divergence regularization (penalize evidence on incorrect classes)
        # alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        alpha_tilde = alpha * (1 - y_onehot) + 1
        
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_reg = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1) \
                 + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=1) \
                 - torch.lgamma(torch.ones_like(S_tilde) * alpha.shape[1]) \
                 + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1)
                 
        loss = nll + lam * kl_reg
        return loss.mean(), nll.mean().detach(), kl_reg.mean().detach()

class PedOSALoss(nn.Module):
    """
    Combined loss for Mamba-PedOSA: Evidential Loss for severity classification,
    Huber loss for AHI regression, and BCE for sleep/wake staging.
    """
    def __init__(self, evidential_coeff=1.0, ahi_coeff=0.5, sw_coeff=0.3):
        super().__init__()
        self.evidential_loss = EvidentialClassificationLoss()
        self.ahi_loss = nn.HuberLoss(delta=2.0)
        self.sw_loss = nn.BCEWithLogitsLoss()
        
        self.evidential_coeff = evidential_coeff
        self.ahi_coeff = ahi_coeff
        self.sw_coeff = sw_coeff
        
    def forward(self, preds, targets, lam=0.1):
        """
        preds: dict with 'alpha', 'ahi_pred', 'sleep_wake_logits'
        targets: dict with 'ahi_label', 'severity_class', 'sleep_wake_label' (optional)
        """
        # 1. Evidential Classification Loss
        alpha = preds['alpha']
        y_cls = targets['severity_class']
        y_onehot = torch.nn.functional.one_hot(y_cls, num_classes=alpha.shape[1]).float()
        
        loss_edl, nll, reg = self.evidential_loss(alpha, y_onehot, lam=lam)
        
        # 2. AHI Regression (Huber with log1p transform)
        y_ahi = targets['ahi_label']
        ahi_pred = preds['ahi_pred']
        loss_ahi = self.ahi_loss(torch.log1p(ahi_pred), torch.log1p(y_ahi))
        
        # 3. Sleep/Wake Staging
        if 'sleep_wake_label' in targets and 'sleep_wake_logits' in preds:
            y_sw = targets['sleep_wake_label'].float()
            loss_sw = self.sw_loss(preds['sleep_wake_logits'], y_sw)
        else:
            loss_sw = torch.tensor(0.0, device=alpha.device)
            
        total_loss = (self.evidential_coeff * loss_edl) + \
                     (self.ahi_coeff * loss_ahi) + \
                     (self.sw_coeff * loss_sw)
            
        return {
            'total_loss': total_loss,
            'evidential_loss': loss_edl,
            'nll_loss': nll,
            'reg_loss': reg,
            'ahi_loss': loss_ahi,
            'sw_loss': loss_sw
        }
