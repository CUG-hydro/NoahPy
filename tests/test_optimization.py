"""
Test module for parameter optimization functionality.

This module tests:
- Parameter initialization
- Physical constraints
- Loss computation
- Gradient computation
- Single optimization step
- Full training loop
- Parameter save/load
"""

import os
import sys
import pytest
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NoahPy.optimization import ParameterOptimizer, PhysicalConstraints
from NoahPy.NoahPy import noah_main


class TestPhysicalConstraints:
    """Test physical constraints for parameters."""

    def test_parameter_bounds(self):
        """Test that parameter bounds are properly defined."""
        assert 'BEXP' in PhysicalConstraints.BOUNDS
        assert 'SMCMAX' in PhysicalConstraints.BOUNDS
        assert 'DKSAT' in PhysicalConstraints.BOUNDS
        assert 'PSISAT' in PhysicalConstraints.BOUNDS

        # Check bounds are reasonable
        for param, (min_val, max_val) in PhysicalConstraints.BOUNDS.items():
            assert min_val < max_val, f"{param} bounds are invalid"
            assert min_val > 0, f"{param} minimum should be positive"

    def test_apply_constraints(self):
        """Test applying constraints to parameters."""
        # Create parameters outside bounds
        bexp = torch.nn.Parameter(torch.tensor([1.0, 15.0], dtype=torch.float32))  # Below and above bounds
        smcmax = torch.nn.Parameter(torch.tensor([0.3, 0.3], dtype=torch.float32))
        dksat = torch.nn.Parameter(torch.tensor([1e-5, 1e-5], dtype=torch.float32))
        psisat = torch.nn.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float32))

        parameters = (bexp, smcmax, dksat, psisat)

        # Apply constraints
        constrained = PhysicalConstraints.apply_constraints(parameters)

        # Check BEXP is within bounds
        bexp_min, bexp_max = PhysicalConstraints.BOUNDS['BEXP']
        assert constrained[0][0].item() >= bexp_min
        assert constrained[0][1].item() <= bexp_max

    def test_validate_parameters(self):
        """Test parameter validation."""
        # Create valid parameters
        bexp = torch.tensor([5.0, 6.0], dtype=torch.float32)
        smcmax = torch.tensor([0.3, 0.35], dtype=torch.float32)
        dksat = torch.tensor([1e-5, 2e-5], dtype=torch.float32)
        psisat = torch.tensor([0.1, 0.15], dtype=torch.float32)

        parameters = (bexp, smcmax, dksat, psisat)

        validation = PhysicalConstraints.validate_parameters(parameters)

        # All should be valid
        assert all(validation.values()), "All parameters should be valid"

        # Create invalid parameters
        bexp_invalid = torch.tensor([100.0], dtype=torch.float32)  # Too large
        parameters_invalid = (bexp_invalid, smcmax[:1], dksat[:1], psisat[:1])

        validation_invalid = PhysicalConstraints.validate_parameters(parameters_invalid)

        assert not validation_invalid['BEXP'], "BEXP should be invalid"


class TestParameterOptimizer:
    """Test ParameterOptimizer class."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = ParameterOptimizer(
            n_layers=10,
            optimizer_type='Adam',
            lr=0.01,
            device='cpu'
        )

        assert optimizer.n_layers == 10
        assert optimizer.optimizer_type == 'Adam'
        assert optimizer.lr == 0.01
        assert optimizer.parameters is None

    def test_parameter_initialization(self):
        """Test parameter initialization."""
        optimizer = ParameterOptimizer(n_layers=5)

        # Initialize with default values
        parameters = optimizer.initialize_parameters(soil_type_index=7)

        assert len(parameters) == 4, "Should have 4 parameter tensors"
        assert all(isinstance(p, torch.nn.Parameter) for p in parameters)
        assert all(p.shape == (5,) for p in parameters), "Each parameter should have n_layers elements"

        # Check parameters are within physical bounds
        validation = PhysicalConstraints.validate_parameters(parameters)
        assert all(validation.values()), "Initialized parameters should be valid"

    def test_parameter_initialization_custom(self):
        """Test parameter initialization with custom values."""
        optimizer = ParameterOptimizer(n_layers=3)

        initial_values = {
            'BEXP': np.array([5.0, 6.0, 7.0]),
            'SMCMAX': np.array([0.3, 0.32, 0.35]),
            'DKSAT': np.array([1e-5, 1.5e-5, 2e-5]),
            'PSISAT': np.array([0.1, 0.12, 0.15]),
        }

        parameters = optimizer.initialize_parameters(initial_values=initial_values)

        # Check values match
        np.testing.assert_array_almost_equal(
            parameters[0].detach().numpy(),
            initial_values['BEXP'],
            decimal=5
        )

    def test_loss_computation(self):
        """Test different loss functions."""
        optimizer = ParameterOptimizer()

        simulated = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        observed = torch.tensor([1.1, 2.1, 2.9, 4.2], dtype=torch.float32)

        # Test MSE
        mse_loss = optimizer.compute_loss(simulated, observed, loss_type='mse')
        expected_mse = torch.mean((simulated - observed) ** 2)
        assert torch.isclose(mse_loss, expected_mse), "MSE loss calculation incorrect"

        # Test MAE
        mae_loss = optimizer.compute_loss(simulated, observed, loss_type='mae')
        expected_mae = torch.mean(torch.abs(simulated - observed))
        assert torch.isclose(mae_loss, expected_mae), "MAE loss calculation incorrect"

        # Test RMSE
        rmse_loss = optimizer.compute_loss(simulated, observed, loss_type='rmse')
        expected_rmse = torch.sqrt(torch.mean((simulated - observed) ** 2))
        assert torch.isclose(rmse_loss, expected_rmse), "RMSE loss calculation incorrect"

    def test_optimizer_setup(self):
        """Test optimizer and scheduler setup."""
        optimizer = ParameterOptimizer(
            n_layers=5,
            optimizer_type='Adam',
            lr=0.01
        )

        parameters = optimizer.initialize_parameters()

        assert optimizer.optimizer is not None
        assert optimizer.scheduler is not None
        assert isinstance(optimizer.optimizer, torch.optim.Adam)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        optimizer = ParameterOptimizer(n_layers=3)
        parameters = optimizer.initialize_parameters()

        # Simple model function: sum of squares
        def simple_model(params, **kwargs):
            return torch.sum(params[0] ** 2)

        # Create dummy observed data
        observed = torch.tensor(0.0)

        # Perform one training step
        loss = optimizer.train_step(simple_model, observed, loss_type='mse')

        # Check gradients exist
        assert parameters[0].grad is not None, "Gradients should be computed"

    def test_parameter_save_load(self, tmp_path):
        """Test saving and loading parameters."""
        optimizer = ParameterOptimizer(n_layers=5)
        parameters = optimizer.initialize_parameters()

        # Add some history
        optimizer.history['train_loss'] = [1.0, 0.9, 0.8]
        optimizer.history['val_loss'] = [1.1, 1.0, 0.9]

        # Save parameters
        save_path = tmp_path / "test_params.pkl"
        optimizer.save_parameters(str(save_path))

        assert save_path.exists(), "Parameter file should be created"

        # Load parameters into new optimizer
        new_optimizer = ParameterOptimizer(n_layers=5)
        new_optimizer.load_parameters(str(save_path))

        # Check parameters match
        for p1, p2 in zip(parameters, new_optimizer.parameters):
            np.testing.assert_array_almost_equal(
                p1.detach().numpy(),
                p2.detach().numpy(),
                decimal=5
            )

        # Check history is preserved
        assert new_optimizer.history['train_loss'] == [1.0, 0.9, 0.8]

    def test_parameter_summary(self):
        """Test parameter summary generation."""
        optimizer = ParameterOptimizer(n_layers=5)
        parameters = optimizer.initialize_parameters()

        summary = optimizer.get_parameter_summary()

        assert 'BEXP' in summary
        assert 'SMCMAX' in summary
        assert 'DKSAT' in summary
        assert 'PSISAT' in summary

        # Check summary contains statistics
        for param_name, stats in summary.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'values' in stats


class TestIntegrationWithNoahPy:
    """Integration tests with NoahPy model."""

    def test_optimization_with_noah_model(self):
        """Test parameter optimization with actual NoahPy model."""
        # Get forcing file path
        forcing_file = os.path.abspath("data/forcing.txt")

        if not os.path.exists(forcing_file):
            pytest.skip(f"Forcing file not found: {forcing_file}")

        # Initialize optimizer
        optimizer = ParameterOptimizer(n_layers=10, lr=0.001)
        parameters = optimizer.initialize_parameters(soil_type_index=7)

        # Define model function wrapper
        def noah_model_wrapper(params, **kwargs):
            """Wrapper to run NoahPy and extract relevant output."""
            # Run model with optimized parameters
            Date, STC, SH2O = noah_main(
                forcing_file,
                trained_parameter=params,
                lstm_model=None,
                output_flag=False
            )
            # Return soil temperature at first layer as output
            return STC[:, 0]  # Shape: (n_timesteps,)

        # Create synthetic "observed" data for testing
        # In real use, this would be actual observations
        with torch.no_grad():
            observed = noah_model_wrapper(parameters)
            # Add some noise to create training target
            observed = observed + torch.randn_like(observed) * 0.1

        # Perform a single training step
        loss_before = optimizer.compute_loss(
            noah_model_wrapper(parameters),
            observed,
            loss_type='mse'
        ).item()

        # Run optimization step
        loss_after = optimizer.train_step(
            noah_model_wrapper,
            observed,
            loss_type='mse'
        )

        # Loss should decrease (though not guaranteed in single step with noise)
        print(f"Loss before: {loss_before:.6f}, Loss after: {loss_after:.6f}")

        # Check that optimization ran without errors
        assert isinstance(loss_after, float)
        assert loss_after >= 0


def test_quick_optimization():
    """Quick test for basic optimization workflow."""
    # This test can be run quickly to verify basic functionality

    optimizer = ParameterOptimizer(n_layers=3, lr=0.1)
    parameters = optimizer.initialize_parameters()

    # Simple quadratic model
    def simple_model(params, **kwargs):
        # Target: parameters close to [5.0, 0.3, 1e-5, 0.1]
        target = torch.tensor([5.0, 0.3, 1e-5, 0.1])
        diff = torch.stack([
            (params[0] - target[0]).mean(),
            (params[1] - target[1]).mean(),
            (params[2] - target[2]).mean(),
            (params[3] - target[3]).mean(),
        ])
        return diff

    observed = torch.zeros(4)

    # Run a few optimization steps
    losses = []
    for i in range(5):
        loss = optimizer.train_step(simple_model, observed, loss_type='mse')
        losses.append(loss)

    # Check that loss decreased
    assert losses[-1] < losses[0], "Loss should decrease during optimization"
    print(f"Losses: {losses}")

    # Check parameter summary
    summary = optimizer.get_parameter_summary()
    print("\nParameter Summary:")
    for param_name, stats in summary.items():
        print(f"{param_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")


if __name__ == "__main__":
    # Run quick test
    print("Running quick optimization test...")
    test_quick_optimization()
    print("\nQuick test passed!")

    # Run all tests with pytest
    print("\nRunning all tests with pytest...")
    pytest.main([__file__, "-v", "-s"])
