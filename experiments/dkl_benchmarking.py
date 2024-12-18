import torch
import torch.nn as nn
import gpytorch
from scipy.io import loadmat
import time
from typing import Tuple, Optional
import numpy as np
from btt import BTTLayer

class DenseFeatureExtractor(nn.Sequential):
    """Traditional dense feature extractor for DKL"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.add_module('linear1', nn.Linear(input_dim, 1000))
        self.add_module('relu1', nn.ReLU())
        self.add_module('linear2', nn.Linear(1000, 500))
        self.add_module('relu2', nn.ReLU())
        self.add_module('linear3', nn.Linear(500, 50))
        self.add_module('relu3', nn.ReLU())
        self.add_module('linear4', nn.Linear(50, 2))

class BTTFeatureExtractor(nn.Sequential):
    """BTT-based feature extractor for DKL"""
    def __init__(self, input_dim: int, tt_rank: int = 8):
        super().__init__()
        self.add_module('btt1', BTTLayer(input_dim, 1000, tt_rank))
        self.add_module('relu1', nn.ReLU())
        self.add_module('btt2', BTTLayer(1000, 500, tt_rank))
        self.add_module('relu2', nn.ReLU())
        self.add_module('btt3', BTTLayer(500, 50, tt_rank))
        self.add_module('relu3', nn.ReLU())
        self.add_module('btt4', BTTLayer(50, 2, tt_rank))

class DKLModel(gpytorch.models.ExactGP):
    """Deep Kernel Learning model with configurable feature extractor"""
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods.Likelihood,
                 feature_extractor: nn.Module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def load_elevators_dataset(test_split: float = 0.2) -> Tuple[torch.Tensor, ...]:
    """Load and preprocess the UCI Elevators dataset"""

    data = torch.Tensor(loadmat('experiments/elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]
    
    train_n = int((1 - test_split) * len(X))
    
    train_x = X[:train_n].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:].contiguous()
    test_y = y[train_n:].contiguous()
    
    return train_x, train_y, test_x, test_y

def train_model(
    model: DKLModel,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_epochs: int = 60,
    learning_rate: float = 0.01,
    device: Optional[torch.device] = None
) -> Tuple[list, float]:
    """Train the DKL model and return metrics"""
    if device is not None:
        model = model.to(device)
        likelihood = likelihood.to(device)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=learning_rate)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    start_time = time.time()
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    training_time = time.time() - start_time
    
    return losses, training_time

def evaluate_model(
    model: DKLModel,
    likelihood: gpytorch.likelihoods.Likelihood,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[float, float, float]:
    """Evaluate model performance and return metrics"""
    if device is not None:
        model = model.to(device)
        likelihood = likelihood.to(device)
        test_x = test_x.to(device)
        test_y = test_y.to(device)
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        start_time = time.time()
        preds = model(test_x)
        inference_time = time.time() - start_time
        
        mae = torch.mean(torch.abs(preds.mean - test_y)).item()
        rmse = torch.sqrt(torch.mean((preds.mean - test_y) ** 2)).item()
        
    return mae, rmse, inference_time

def run_experiment(tt_ranks=[4, 8, 16], n_epochs=60, device=None):
    """Run complete experiment comparing Dense vs BTT implementations"""
    # Load dataset
    train_x, train_y, test_x, test_y = load_elevators_dataset()
    input_dim = train_x.size(-1)
    
    results = {
        'dense': {},
        'btt': {rank: {} for rank in tt_ranks}
    }
    
    # Baseline Dense Model
    print("Training Dense DKL model...")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    dense_model = DKLModel(
        train_x, train_y,
        likelihood,
        DenseFeatureExtractor(input_dim)
    )
    
    losses, train_time = train_model(
        dense_model, likelihood,
        train_x, train_y,
        n_epochs=n_epochs,
        device=device
    )
    
    mae, rmse, inf_time = evaluate_model(
        dense_model, likelihood,
        test_x, test_y,
        device=device
    )
    
    results['dense'] = {
        'training_time': train_time,
        'inference_time': inf_time,
        'mae': mae,
        'rmse': rmse,
        'final_loss': losses[-1],
        'losses': losses
    }
    
    # BTT Models with different ranks
    for rank in tt_ranks:
        print(f"Training BTT DKL model with rank {rank}...")
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        btt_model = DKLModel(
            train_x, train_y,
            likelihood,
            BTTFeatureExtractor(input_dim, tt_rank=rank)
        )
        
        losses, train_time = train_model(
            btt_model, likelihood,
            train_x, train_y,
            n_epochs=n_epochs,
            device=device
        )
        
        mae, rmse, inf_time = evaluate_model(
            btt_model, likelihood,
            test_x, test_y,
            device=device
        )
        
        results['btt'][rank] = {
            'training_time': train_time,
            'inference_time': inf_time,
            'mae': mae,
            'rmse': rmse,
            'final_loss': losses[-1],
            'losses': losses
        }
    
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    results = run_experiment(
        tt_ranks=[2, 4, 8, 16],
        n_epochs=60,
        device=device
    )
    
    # Print results summary
    print("\nResults Summary:")
    print("-" * 50)
    
    print("\nDense DKL Model:")
    print(f"Training time: {results['dense']['training_time']:.2f}s")
    print(f"Inference time: {results['dense']['inference_time']:.2f}s")
    print(f"MAE: {results['dense']['mae']:.4f}")
    print(f"RMSE: {results['dense']['rmse']:.4f}")
    
    for rank, metrics in results['btt'].items():
        print(f"\nBTT DKL Model (rank={rank}):")
        print(f"Training time: {metrics['training_time']:.2f}s")
        print(f"Inference time: {metrics['inference_time']:.2f}s")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")