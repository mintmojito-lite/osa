import torch
import torch.nn as nn

class EvidentialRegressionLoss(nn.Module):
    """
    Negative Log-Likelihood of Normal-Inverse-Gamma distribution
    with regularization to penalize overconfidence on incorrect predictions.
    """
    def __init__(self):
        super().__init__()
        
    def nig_nll(self, y, gamma, nu, alpha, beta):
        # Negative log-likelihood of Normal-Inverse-Gamma
        omega = 2 * beta * (1 + nu)
        nll = (0.5 * torch.log(torch.pi / nu)
               - alpha * torch.log(omega)
               + (alpha + 0.5) * torch.log((y - gamma)**2 * nu + omega)
               + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
        return nll.mean()
        
    def nig_regularizer(self, y, gamma, nu, alpha):
        # Penalize overconfidence on wrong predictions
        return (torch.abs(y - gamma) * (2 * nu + alpha)).mean()
        
    def forward(self, y, gamma, nu, alpha, beta, lam=0.1):
        nll = self.nig_nll(y, gamma, nu, alpha, beta)
        reg = self.nig_regularizer(y, gamma, nu, alpha)
        return nll + lam * reg, nll.detach(), reg.detach()

class PedOSALoss(nn.Module):
    """
    Combined loss for Mamba-PedOSA: Evidential Loss for AHI regression
    and Focal Loss / Cross Entropy for severity classification.
    """
    def __init__(self, evidential_coeff=0.1, cls_weight=1.0, focal_weights=None):
        super().__init__()
        self.evidential_loss = EvidentialRegressionLoss()
        
        if focal_weights is not None:
            # We assume it's a tensor passed from lightning module
            self.cls_loss = nn.CrossEntropyLoss(weight=focal_weights)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
            
        self.cls_weight = cls_weight
        
    def forward(self, preds, targets, lam=0.1):
        """
        preds: dict with 'gamma', 'nu', 'alpha', 'beta', 'class_logits' (optional)
        targets: dict with 'ahi_label', 'severity_class' (optional)
        """
        y = targets['ahi_label']
        
        gamma = preds['gamma']
        nu = preds['nu']
        alpha = preds['alpha']
        beta = preds['beta']
        
        loss_reg, nll, reg = self.evidential_loss(y, gamma, nu, alpha, beta, lam=lam)
        
        if 'class_logits' in preds and 'severity_class' in targets:
            y_cls = targets['severity_class']
            loss_cls = self.cls_loss(preds['class_logits'], y_cls)
            total_loss = loss_reg + self.cls_weight * loss_cls
        else:
            loss_cls = torch.tensor(0.0, device=y.device)
            total_loss = loss_reg
            
        return {
            'total_loss': total_loss,
            'evidential_loss': loss_reg,
            'nll_loss': nll,
            'reg_loss': reg,
            'cls_loss': loss_cls
        }
