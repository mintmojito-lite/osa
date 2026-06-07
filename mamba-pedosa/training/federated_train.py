import os
import argparse
import flwr as fl
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import yaml

from data.dataset import get_dataloaders
from training.lightning_module import MambaPedOSAModule

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train for 3 epochs per round as per paper (Section 5.6)
        trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices=1, enable_progress_bar=False, logger=False)
        trainer.fit(self.model, self.train_loader)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        trainer = pl.Trainer(accelerator="auto", devices=1, enable_progress_bar=False, logger=False)
        results = trainer.validate(self.model, self.val_loader)
        loss = results[0]["val_loss"]
        return float(loss), len(self.val_loader.dataset), {"val_loss": float(loss)}

def client_fn(cid: str) -> FlowerClient:
    # Load model configuration
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset for the specific client (simulating BCH, NCH, CHAT)
    # In a real setup, we would partition the data based on `cid`
    train_loader, val_loader, _, _ = get_dataloaders('data/features')
    
    model = MambaPedOSAModule(config)
    return FlowerClient(model, train_loader, val_loader)

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Pilot with Flower")
    parser.add_argument("--num_rounds", type=int, default=30, help="Number of federated learning rounds")
    args = parser.parse_args()

    print(f"Starting Federated Learning Pilot for {args.num_rounds} rounds across 3 nodes...")
    
    # Define strategy (Federated Averaging)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    # Start Simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3, # BCH, NCH, CHAT
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
