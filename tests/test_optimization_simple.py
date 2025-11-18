"""
Simple test script for parameter optimization (no pytest required).

This script tests basic functionality of the optimization module.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from NoahPy.optimization import ParameterOptimizer, PhysicalConstraints
from NoahPy.NoahPy import noah_main


def test_parameter_initialization():
    """Test basic parameter initialization."""
    print("\n" + "="*70)
    print("TEST 1: Parameter Initialization")
    print("="*70)

    optimizer = ParameterOptimizer(n_layers=5, lr=0.01)
    parameters = optimizer.initialize_parameters(soil_type_index=7)

    print(f"✓ Initialized {len(parameters)} parameter tensors")
    print(f"✓ Each parameter has shape: {parameters[0].shape}")

    # Check validity
    validation = PhysicalConstraints.validate_parameters(parameters)
    print(f"✓ Parameter validation: {validation}")

    assert all(validation.values()), "All parameters should be valid"
    print("✓ TEST PASSED: Parameter initialization")

    return optimizer, parameters


def test_physical_constraints():
    """Test physical constraints."""
    print("\n" + "="*70)
    print("TEST 2: Physical Constraints")
    print("="*70)

    # Create parameters outside bounds
    bexp = torch.nn.Parameter(torch.tensor([1.0, 15.0], dtype=torch.float32))
    smcmax = torch.nn.Parameter(torch.tensor([0.3, 0.3], dtype=torch.float32))
    dksat = torch.nn.Parameter(torch.tensor([1e-5, 1e-5], dtype=torch.float32))
    psisat = torch.nn.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float32))

    print(f"Before constraint - BEXP: {bexp.data.numpy()}")

    parameters = (bexp, smcmax, dksat, psisat)
    constrained = PhysicalConstraints.apply_constraints(parameters)

    print(f"After constraint  - BEXP: {constrained[0].data.numpy()}")

    bexp_min, bexp_max = PhysicalConstraints.BOUNDS['BEXP']
    assert constrained[0][0].item() >= bexp_min, "BEXP[0] should be >= minimum"
    assert constrained[0][1].item() <= bexp_max, "BEXP[1] should be <= maximum"

    print("✓ TEST PASSED: Physical constraints applied correctly")


def test_loss_computation():
    """Test loss functions."""
    print("\n" + "="*70)
    print("TEST 3: Loss Computation")
    print("="*70)

    optimizer = ParameterOptimizer()

    simulated = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    observed = torch.tensor([1.1, 2.1, 2.9, 4.2], dtype=torch.float32)

    # Test MSE
    mse_loss = optimizer.compute_loss(simulated, observed, loss_type='mse')
    expected_mse = torch.mean((simulated - observed) ** 2)

    print(f"MSE Loss: {mse_loss.item():.6f}")
    print(f"Expected: {expected_mse.item():.6f}")
    assert torch.isclose(mse_loss, expected_mse), "MSE calculation error"

    # Test MAE
    mae_loss = optimizer.compute_loss(simulated, observed, loss_type='mae')
    print(f"MAE Loss: {mae_loss.item():.6f}")

    # Test RMSE
    rmse_loss = optimizer.compute_loss(simulated, observed, loss_type='rmse')
    print(f"RMSE Loss: {rmse_loss.item():.6f}")

    print("✓ TEST PASSED: Loss computation")


def test_simple_optimization():
    """Test basic optimization loop."""
    print("\n" + "="*70)
    print("TEST 4: Simple Optimization Loop")
    print("="*70)

    optimizer = ParameterOptimizer(n_layers=3, lr=0.1, clip_grad=1.0)
    parameters = optimizer.initialize_parameters()

    print("Initial parameters:")
    for i, param in enumerate(parameters):
        print(f"  Param {i}: {param.data.numpy()}")

    # Simple quadratic model - try to move parameters toward target
    target_values = torch.tensor([5.0, 0.3, 1e-5, 0.1])

    def simple_model(params, **kwargs):
        """Simple model that returns distance from target."""
        diff = torch.stack([
            (params[0] - target_values[0]).mean(),
            (params[1] - target_values[1]).mean(),
            (params[2] - target_values[2]).mean(),
            (params[3] - target_values[3]).mean(),
        ])
        return diff

    observed = torch.zeros(4)

    # Run optimization steps
    print("\nOptimization progress:")
    losses = []
    for i in range(10):
        loss = optimizer.train_step(simple_model, observed, loss_type='mse')
        losses.append(loss)
        if i % 2 == 0:
            print(f"  Step {i:2d}: Loss = {loss:.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")

    assert losses[-1] < losses[0], "Loss should decrease"
    print("✓ TEST PASSED: Optimization reduces loss")

    return optimizer


def test_parameter_summary():
    """Test parameter summary."""
    print("\n" + "="*70)
    print("TEST 5: Parameter Summary")
    print("="*70)

    optimizer = ParameterOptimizer(n_layers=5)
    parameters = optimizer.initialize_parameters()

    summary = optimizer.get_parameter_summary()

    print("\nParameter Statistics:")
    for param_name, stats in summary.items():
        print(f"\n{param_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std:  {stats['std']:.6f}")
        print(f"  Min:  {stats['min']:.6f}")
        print(f"  Max:  {stats['max']:.6f}")

    assert 'BEXP' in summary
    assert 'mean' in summary['BEXP']
    print("\n✓ TEST PASSED: Parameter summary generation")


def test_save_load_parameters():
    """Test saving and loading parameters."""
    print("\n" + "="*70)
    print("TEST 6: Save and Load Parameters")
    print("="*70)

    # Create optimizer and add some history
    optimizer1 = ParameterOptimizer(n_layers=3)
    params1 = optimizer1.initialize_parameters()
    optimizer1.history['train_loss'] = [1.0, 0.9, 0.8]

    # Save parameters
    save_path = "/tmp/test_params.pkl"
    optimizer1.save_parameters(save_path)
    print(f"✓ Parameters saved to {save_path}")

    # Load into new optimizer
    optimizer2 = ParameterOptimizer(n_layers=3)
    optimizer2.load_parameters(save_path)
    print(f"✓ Parameters loaded from {save_path}")

    # Compare
    for i, (p1, p2) in enumerate(zip(params1, optimizer2.parameters)):
        diff = torch.abs(p1 - p2).max().item()
        print(f"  Param {i} max difference: {diff:.10f}")
        assert diff < 1e-5, f"Parameters {i} don't match"

    assert optimizer2.history['train_loss'] == [1.0, 0.9, 0.8]
    print("✓ TEST PASSED: Save/load preserves parameters and history")

    # Cleanup
    os.remove(save_path)


def test_with_noah_model():
    """Test with actual NoahPy model."""
    print("\n" + "="*70)
    print("TEST 7: Integration with NoahPy Model")
    print("="*70)

    forcing_file = os.path.abspath("data/forcing.txt")

    if not os.path.exists(forcing_file):
        print(f"⚠ Skipping: forcing file not found at {forcing_file}")
        return

    print(f"Using forcing file: {forcing_file}")

    # Initialize optimizer
    optimizer = ParameterOptimizer(n_layers=10, lr=0.001)
    parameters = optimizer.initialize_parameters(soil_type_index=7)

    print("\nRunning NoahPy model with optimized parameters...")

    # Define model wrapper
    def noah_model_wrapper(params, **kwargs):
        """Run NoahPy and return soil temperature."""
        Date, STC, SH2O = noah_main(
            forcing_file,
            trained_parameter=params,
            lstm_model=None,
            output_flag=False
        )
        # Return first layer soil temperature
        return STC[:, 0]

    # Get baseline output
    print("Computing baseline simulation...")
    with torch.no_grad():
        baseline_output = noah_model_wrapper(parameters)
        print(f"✓ Baseline output shape: {baseline_output.shape}")
        print(f"  Mean temperature: {baseline_output.mean().item():.2f} K")
        print(f"  Temperature range: [{baseline_output.min().item():.2f}, {baseline_output.max().item():.2f}] K")

        # Create synthetic observed data
        observed = baseline_output + torch.randn_like(baseline_output) * 0.5

    # Test single optimization step
    print("\nPerforming optimization step...")
    loss_before = optimizer.compute_loss(
        noah_model_wrapper(parameters),
        observed,
        loss_type='mse'
    ).item()

    loss_after = optimizer.train_step(
        noah_model_wrapper,
        observed,
        loss_type='mse'
    )

    print(f"Loss before step: {loss_before:.6f}")
    print(f"Loss after step:  {loss_after:.6f}")

    assert isinstance(loss_after, float)
    assert loss_after >= 0

    print("✓ TEST PASSED: NoahPy integration successful")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# NoahPy Parameter Optimization Test Suite")
    print("#"*70)

    tests = [
        ("Parameter Initialization", test_parameter_initialization),
        ("Physical Constraints", test_physical_constraints),
        ("Loss Computation", test_loss_computation),
        ("Simple Optimization", test_simple_optimization),
        ("Parameter Summary", test_parameter_summary),
        ("Save/Load Parameters", test_save_load_parameters),
        ("NoahPy Integration", test_with_noah_model),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "#"*70)
    print(f"# Test Summary: {passed} passed, {failed} failed")
    print("#"*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
