import torch
import torch.nn as nn
import torch.nn.functional as F

class DERHead(nn.Module):
    """
    Deep Evidential Regression Head.
    Outputs the 4 parameters of the Normal-Inverse-Gamma distribution:
    gamma (mu), nu (v), alpha, beta.
    Also outputs logits for auxiliary severity classification (4 classes).
    """
    def __init__(self, in_features=256, hidden_dim=512, dropout=0.1, num_classes=4):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Evidential regression output: 4 parameters
        self.evidential_out = nn.Linear(hidden_dim, 4)
        
        # Auxiliary classification output
        self.class_out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        x: (B, in_features)
        Returns:
            mu, v, alpha, beta, class_logits
        """
        x = self.dropout(self.gelu(self.bn1(self.fc1(x))))
        
        ev_params = self.evidential_out(x)
        
        # Transform parameters to satisfy constraints:
        # mu: no constraint
        # v > 0
        # alpha > 1
        # beta > 0
        
        mu = ev_params[:, 0]
        v = F.softplus(ev_params[:, 1]) + 1e-6
        alpha = F.softplus(ev_params[:, 2]) + 1.0 + 1e-6
        beta = F.softplus(ev_params[:, 3]) + 1e-6
        
        class_logits = self.class_out(x)
        
        return mu, v, alpha, beta, class_logits
