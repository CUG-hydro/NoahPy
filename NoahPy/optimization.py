'''
@Project ：NoahPy
@File    ：optimization.py
@Author  ：Claude Code
@Date    ：2025-01-18

@Description：
Parameter optimization module for NoahPy. This module provides tools for calibrating soil parameters
using gradient-based optimization methods. It supports:
- Parameter initialization and constraints
- Multiple loss functions (MSE, MAE, custom)
- Training and validation loops
- Parameter persistence (save/load)
- Gradient clipping and early stopping

@License：
Copyright (c) 2025 NoahPy Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction.
'''

import os
import pickle
from typing import Optional, Tuple, Dict, List, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class PhysicalConstraints:
    """
    Physical constraints for soil parameters to ensure physically realistic values.
    Based on typical soil parameter ranges from SOILPARM.TBL.
    """

    # Parameter bounds: (min, max) based on SOILPARM.TBL ranges
    BOUNDS = {
        'BEXP': (2.7, 12.0),      # Pore size distribution index
        'SMCMAX': (0.13, 0.50),   # Saturated soil moisture content (m³/m³)
        'DKSAT': (1e-7, 2e-4),    # Saturated hydraulic conductivity (m/s)
        'PSISAT': (0.03, 0.80),   # Saturated matric potential (m)
    }

    @staticmethod
    def apply_constraints(parameters: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Apply physical constraints to parameters by clamping to valid ranges.

        Args:
            parameters: Tuple of (BEXP, SMCMAX, DKSAT, PSISAT) tensors

        Returns:
            Constrained parameters tuple
        """
        param_names = ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']
        constrained = []

        with torch.no_grad():
            for param, name in zip(parameters, param_names):
                min_val, max_val = PhysicalConstraints.BOUNDS[name]
                param.data.clamp_(min_val, max_val)
                constrained.append(param)

        return tuple(constrained)

    @staticmethod
    def validate_parameters(parameters: Tuple[torch.Tensor, ...]) -> Dict[str, bool]:
        """
        Validate parameters against physical constraints.

        Args:
            parameters: Tuple of (BEXP, SMCMAX, DKSAT, PSISAT) tensors

        Returns:
            Dictionary with validation results for each parameter
        """
        param_names = ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']
        validation = {}

        for param, name in zip(parameters, param_names):
            min_val, max_val = PhysicalConstraints.BOUNDS[name]
            is_valid = (param >= min_val).all() and (param <= max_val).all()
            validation[name] = is_valid.item()

        return validation


class ParameterOptimizer:
    """
    Main optimizer class for NoahPy parameter calibration.

    This class manages the optimization process including:
    - Parameter initialization
    - Loss computation
    - Training/validation loops
    - Parameter saving/loading
    - Gradient clipping and scheduling
    """

    def __init__(
        self,
        n_layers: int = 10,
        optimizer_type: str = 'Adam',
        lr: float = 0.01,
        weight_decay: float = 0.0,
        clip_grad: Optional[float] = 1.0,
        device: str = 'cpu'
    ):
        """
        Initialize the parameter optimizer.

        Args:
            n_layers: Number of soil layers to optimize (default: 10, top layers)
            optimizer_type: Type of optimizer ('Adam', 'AdamW', 'SGD')
            lr: Learning rate
            weight_decay: Weight decay for regularization
            clip_grad: Maximum gradient norm for clipping (None to disable)
            device: Device to use ('cpu' or 'cuda')
        """
        self.n_layers = n_layers
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.device = torch.device(device)

        # Initialize parameters
        self.parameters = None
        self.optimizer = None
        self.scheduler = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }

    def initialize_parameters(
        self,
        initial_values: Optional[Dict[str, np.ndarray]] = None,
        soil_type_index: int = 7
    ) -> Tuple[torch.nn.Parameter, ...]:
        """
        Initialize soil parameters for optimization.

        Args:
            initial_values: Dictionary with initial parameter values
                           Keys: 'BEXP', 'SMCMAX', 'DKSAT', 'PSISAT'
                           Values: numpy arrays of shape (n_layers,)
            soil_type_index: Soil type index for default initialization (if initial_values is None)

        Returns:
            Tuple of torch.nn.Parameter objects (BEXP, SMCMAX, DKSAT, PSISAT)
        """
        if initial_values is None:
            # Use default values from SOILPARM.TBL for the specified soil type
            from .Module_sf_noahlsm import Soil_Parameter

            stpnum = torch.full((self.n_layers,), soil_type_index, dtype=torch.long)
            BEXP, SMCMAX, DKSAT, PSISAT = Soil_Parameter.get_by_index(stpnum)

            # Convert to numpy for initialization
            initial_values = {
                'BEXP': BEXP.detach().cpu().numpy(),
                'SMCMAX': SMCMAX.detach().cpu().numpy(),
                'DKSAT': DKSAT.detach().cpu().numpy(),
                'PSISAT': PSISAT.detach().cpu().numpy(),
            }

        # Create learnable parameters
        self.parameters = (
            nn.Parameter(torch.tensor(initial_values['BEXP'], dtype=torch.float32, device=self.device)),
            nn.Parameter(torch.tensor(initial_values['SMCMAX'], dtype=torch.float32, device=self.device)),
            nn.Parameter(torch.tensor(initial_values['DKSAT'], dtype=torch.float32, device=self.device)),
            nn.Parameter(torch.tensor(initial_values['PSISAT'], dtype=torch.float32, device=self.device)),
        )

        # Apply initial constraints
        self.parameters = PhysicalConstraints.apply_constraints(self.parameters)

        # Initialize optimizer
        self._setup_optimizer()

        return self.parameters

    def _setup_optimizer(self):
        """Setup the optimizer and learning rate scheduler."""
        if self.parameters is None:
            raise ValueError("Parameters must be initialized before setting up optimizer")

        # Create optimizer
        if self.optimizer_type == 'Adam':
            self.optimizer = Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'AdamW':
            self.optimizer = AdamW(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = SGD(self.parameters, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def compute_loss(
        self,
        simulated: torch.Tensor,
        observed: torch.Tensor,
        loss_type: str = 'mse',
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss between simulated and observed values.

        Args:
            simulated: Simulated values from model
            observed: Observed values (ground truth)
            loss_type: Type of loss ('mse', 'mae', 'rmse', 'nse')
            weights: Optional weights for each time step

        Returns:
            Loss value as a scalar tensor
        """
        if weights is None:
            weights = torch.ones_like(simulated)

        if loss_type == 'mse':
            loss = torch.mean(weights * (simulated - observed) ** 2)
        elif loss_type == 'mae':
            loss = torch.mean(weights * torch.abs(simulated - observed))
        elif loss_type == 'rmse':
            loss = torch.sqrt(torch.mean(weights * (simulated - observed) ** 2))
        elif loss_type == 'nse':
            # Nash-Sutcliffe Efficiency (as a loss: 1 - NSE)
            numerator = torch.sum(weights * (observed - simulated) ** 2)
            denominator = torch.sum(weights * (observed - torch.mean(observed)) ** 2)
            nse = 1 - numerator / (denominator + 1e-8)
            loss = 1 - nse  # Convert to loss (minimize)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return loss

    def train_step(
        self,
        model_fn: Callable,
        observed_data: torch.Tensor,
        loss_type: str = 'mse',
        **model_kwargs
    ) -> float:
        """
        Perform a single training step.

        Args:
            model_fn: Function that runs the model and returns simulated values
                     Should accept parameters as first argument: model_fn(parameters, **kwargs)
            observed_data: Observed data for comparison
            loss_type: Type of loss function
            **model_kwargs: Additional keyword arguments for model_fn

        Returns:
            Loss value as a Python float
        """
        self.optimizer.zero_grad()

        # Forward pass
        simulated = model_fn(self.parameters, **model_kwargs)

        # Compute loss
        loss = self.compute_loss(simulated, observed_data, loss_type=loss_type)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.clip_grad)

        # Optimization step
        self.optimizer.step()

        # Apply physical constraints
        self.parameters = PhysicalConstraints.apply_constraints(self.parameters)

        return loss.item()

    def validate(
        self,
        model_fn: Callable,
        observed_data: torch.Tensor,
        loss_type: str = 'mse',
        **model_kwargs
    ) -> float:
        """
        Perform validation without updating parameters.

        Args:
            model_fn: Function that runs the model and returns simulated values
            observed_data: Observed data for comparison
            loss_type: Type of loss function
            **model_kwargs: Additional keyword arguments for model_fn

        Returns:
            Validation loss as a Python float
        """
        with torch.no_grad():
            simulated = model_fn(self.parameters, **model_kwargs)
            loss = self.compute_loss(simulated, observed_data, loss_type=loss_type)

        return loss.item()

    def train(
        self,
        model_fn: Callable,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        n_epochs: int = 100,
        loss_type: str = 'mse',
        early_stop_patience: int = 10,
        verbose: bool = True,
        **model_kwargs
    ) -> Dict[str, List[float]]:
        """
        Full training loop with validation and early stopping.

        Args:
            model_fn: Function that runs the model
            train_data: Training data (observed values)
            val_data: Validation data (optional)
            n_epochs: Number of training epochs
            loss_type: Type of loss function
            early_stop_patience: Number of epochs to wait before early stopping
            verbose: Whether to print training progress
            **model_kwargs: Additional keyword arguments for model_fn

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training step
            train_loss = self.train_step(model_fn, train_data, loss_type=loss_type, **model_kwargs)
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Validation step
            if val_data is not None:
                val_loss = self.validate(model_fn, val_data, loss_type=loss_type, **model_kwargs)
                self.history['val_loss'].append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{n_epochs} - "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{n_epochs} - "
                          f"Train Loss: {train_loss:.6f}, "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        return self.history

    def save_parameters(self, filepath: str):
        """
        Save optimized parameters to file.

        Args:
            filepath: Path to save parameters
        """
        if self.parameters is None:
            raise ValueError("No parameters to save")

        param_dict = {
            'BEXP': self.parameters[0].detach().cpu().numpy(),
            'SMCMAX': self.parameters[1].detach().cpu().numpy(),
            'DKSAT': self.parameters[2].detach().cpu().numpy(),
            'PSISAT': self.parameters[3].detach().cpu().numpy(),
            'n_layers': self.n_layers,
            'history': self.history,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(param_dict, f)

        print(f"Parameters saved to {filepath}")

    def load_parameters(self, filepath: str):
        """
        Load parameters from file.

        Args:
            filepath: Path to parameter file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Parameter file not found: {filepath}")

        with open(filepath, 'rb') as f:
            param_dict = pickle.load(f)

        # Initialize parameters with loaded values
        initial_values = {
            'BEXP': param_dict['BEXP'],
            'SMCMAX': param_dict['SMCMAX'],
            'DKSAT': param_dict['DKSAT'],
            'PSISAT': param_dict['PSISAT'],
        }

        self.n_layers = param_dict['n_layers']
        self.history = param_dict.get('history', {'train_loss': [], 'val_loss': [], 'lr': []})

        self.initialize_parameters(initial_values)

        print(f"Parameters loaded from {filepath}")

    def get_parameter_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of current parameter values.

        Returns:
            Dictionary with statistics for each parameter
        """
        if self.parameters is None:
            raise ValueError("Parameters not initialized")

        param_names = ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']
        summary = {}

        for param, name in zip(self.parameters, param_names):
            values = param.detach().cpu().numpy()
            summary[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values.tolist(),
            }

        return summary


def create_optimizer_from_config(config: Dict) -> ParameterOptimizer:
    """
    Create a ParameterOptimizer from a configuration dictionary.

    Args:
        config: Configuration dictionary with optimizer settings

    Returns:
        Configured ParameterOptimizer instance

    Example:
        config = {
            'n_layers': 10,
            'optimizer_type': 'Adam',
            'lr': 0.01,
            'weight_decay': 1e-5,
            'clip_grad': 1.0,
            'device': 'cpu'
        }
        optimizer = create_optimizer_from_config(config)
    """
    return ParameterOptimizer(**config)
