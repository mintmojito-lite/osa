import torch
import torch.nn as nn
import torch.nn.functional as F

class DERHead(nn.Module):
    """
    Deep Evidential Regression Head.
    Outputs the parameters of a Normal Inverse-Gamma distribution:
    gamma (mu), nu (v), alpha, beta.
    Calculates Aleatoric and Epistemic uncertainties and Evidence score S.
    """
    def __init__(self, in_features=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 4)
        )
        
    def forward(self, x):
        """
        x: (B, D)
        """
        out = self.net(x)
        
        gamma = out[:, 0]
        log_nu = out[:, 1]
        log_alpha = out[:, 2]
        log_beta = out[:, 3]
        
        # Apply constraints
        nu = F.softplus(log_nu) + 1e-6
        alpha = F.softplus(log_alpha) + 1.0 + 1e-6
        beta = F.softplus(log_beta) + 1e-6
        
        # Calculate uncertainties
        # Aleatoric uncertainty: beta / (alpha - 1)
        aleatoric = beta / (alpha - 1.0)
        
        # Epistemic uncertainty: beta / (nu * (alpha - 1))
        epistemic = beta / (nu * (alpha - 1.0))
        
        # Evidence score S: nu + 2*alpha
        evidence_score = nu + 2 * alpha
        
        return {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'evidence_score': evidence_score
        }
