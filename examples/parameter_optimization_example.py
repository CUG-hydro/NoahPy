"""
Parameter Optimization Example for NoahPy

This script demonstrates how to use the parameter optimization module to calibrate
soil parameters using observed data.

Usage:
    python examples/parameter_optimization_example.py

Requirements:
    - forcing.txt in data/ directory
    - Observed soil temperature or moisture data (optional, will use synthetic data if not provided)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NoahPy.NoahPy import noah_main
from NoahPy.optimization import ParameterOptimizer, PhysicalConstraints


def prepare_observed_data(forcing_file, noise_level=0.5):
    """
    Prepare observed data for optimization.

    In a real application, this would load actual observations.
    For demonstration, we create synthetic data by running the model
    with default parameters and adding noise.

    Args:
        forcing_file: Path to forcing data file
        noise_level: Standard deviation of Gaussian noise to add (K)

    Returns:
        Observed soil temperature tensor
    """
    print("Generating synthetic observed data...")

    # Run model with default parameters
    Date, STC, SH2O = noah_main(
        forcing_file,
        trained_parameter=None,
        lstm_model=None,
        output_flag=False
    )

    # Use first layer soil temperature as "observations"
    # In reality, you would load actual field measurements here
    observed = STC[:, 0].clone()

    # Add noise to simulate observation uncertainty
    observed = observed + torch.randn_like(observed) * noise_level

    print(f"  Generated {len(observed)} time steps of synthetic observations")
    print(f"  Temperature range: [{observed.min():.2f}, {observed.max():.2f}] K")

    return observed, Date


def run_optimization_example():
    """Run a complete parameter optimization example."""

    print("\n" + "="*70)
    print("NoahPy Parameter Optimization Example")
    print("="*70 + "\n")

    # Configuration
    forcing_file = os.path.abspath("data/forcing.txt")
    n_epochs = 50
    learning_rate = 0.001
    n_layers = 10  # Optimize top 10 soil layers

    if not os.path.exists(forcing_file):
        print(f"Error: Forcing file not found at {forcing_file}")
        print("Please ensure data/forcing.txt exists.")
        return

    print(f"Configuration:")
    print(f"  Forcing file: {forcing_file}")
    print(f"  Number of epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Layers to optimize: {n_layers}")
    print()

    # Step 1: Prepare observed data
    observed_data, dates = prepare_observed_data(forcing_file, noise_level=0.5)

    # Split into training and validation sets
    split_idx = int(len(observed_data) * 0.8)
    train_data = observed_data[:split_idx]
    val_data = observed_data[split_idx:]

    print(f"\nData split:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print()

    # Step 2: Initialize optimizer
    print("Initializing parameter optimizer...")
    optimizer = ParameterOptimizer(
        n_layers=n_layers,
        optimizer_type='Adam',
        lr=learning_rate,
        weight_decay=1e-5,
        clip_grad=1.0,
        device='cpu'
    )

    # Initialize parameters with default values for soil type 7
    parameters = optimizer.initialize_parameters(soil_type_index=7)

    print("Initial parameter values:")
    summary = optimizer.get_parameter_summary()
    for param_name in ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']:
        print(f"  {param_name:8s}: mean={summary[param_name]['mean']:10.6f}, "
              f"range=[{summary[param_name]['min']:.6f}, {summary[param_name]['max']:.6f}]")
    print()

    # Step 3: Define model wrapper
    def noah_model_wrapper(params, use_train_data=True, **kwargs):
        """
        Wrapper function to run NoahPy and extract simulated values.

        Args:
            params: Optimized soil parameters
            use_train_data: Whether to use training data subset

        Returns:
            Simulated soil temperature for first layer
        """
        Date, STC, SH2O = noah_main(
            forcing_file,
            trained_parameter=params,
            lstm_model=None,
            output_flag=False
        )

        # Extract first layer temperature
        sim_temp = STC[:, 0]

        # Return appropriate subset
        if use_train_data:
            return sim_temp[:split_idx]
        else:
            return sim_temp[split_idx:]

    # Step 4: Run optimization
    print("Starting parameter optimization...\n")

    # Manual training loop with custom logic
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training step
        train_loss = optimizer.train_step(
            noah_model_wrapper,
            train_data,
            loss_type='mse',
            use_train_data=True
        )
        train_losses.append(train_loss)

        # Validation step
        val_loss = optimizer.validate(
            noah_model_wrapper,
            val_data,
            loss_type='mse',
            use_train_data=False
        )
        val_losses.append(val_loss)

        # Update learning rate based on validation loss
        optimizer.scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {optimizer.optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\nOptimization completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print()

    # Step 5: Display optimized parameters
    print("Optimized parameter values:")
    final_summary = optimizer.get_parameter_summary()
    for param_name in ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']:
        initial = summary[param_name]['mean']
        final = final_summary[param_name]['mean']
        change = ((final - initial) / initial) * 100
        print(f"  {param_name:8s}: {initial:10.6f} → {final:10.6f} "
              f"(change: {change:+6.2f}%)")
    print()

    # Step 6: Validate parameters
    validation = PhysicalConstraints.validate_parameters(optimizer.parameters)
    print("Parameter validation:")
    for param_name, is_valid in validation.items():
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"  {param_name:8s}: {status}")
    print()

    # Step 7: Save optimized parameters
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    param_file = os.path.join(output_dir, "optimized_parameters.pkl")
    optimizer.save_parameters(param_file)
    print(f"Optimized parameters saved to: {param_file}\n")

    # Step 8: Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Training and validation loss
    ax = axes[0, 0]
    ax.plot(train_losses, label='Training Loss', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Parameter evolution (BEXP as example)
    # Note: Would need to track parameter history during training
    # For now, show initial vs final
    ax = axes[0, 1]
    param_names = ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']
    initial_values = [summary[p]['mean'] for p in param_names]
    final_values = [final_summary[p]['mean'] for p in param_names]

    x = np.arange(len(param_names))
    width = 0.35
    ax.bar(x - width/2, initial_values, width, label='Initial', alpha=0.8)
    ax.bar(x + width/2, final_values, width, label='Optimized', alpha=0.8)
    ax.set_ylabel('Normalized Value')
    ax.set_title('Parameter Changes')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Simulated vs Observed (validation set)
    ax = axes[1, 0]
    with torch.no_grad():
        sim_val = noah_model_wrapper(optimizer.parameters, use_train_data=False)

    time_steps = np.arange(len(val_data))
    ax.plot(time_steps, val_data.numpy(), 'o', label='Observed', alpha=0.6, markersize=3)
    ax.plot(time_steps, sim_val.numpy(), '-', label='Simulated', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Soil Temperature (K)')
    ax.set_title('Validation: Simulated vs Observed')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Residuals
    ax = axes[1, 1]
    residuals = (sim_val - val_data).numpy()
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (K)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution (Mean: {residuals.mean():.4f} K)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    fig_file = os.path.join(output_dir, "optimization_results.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_file}")

    # Optionally display
    # plt.show()

    print("\n" + "="*70)
    print("Parameter optimization example completed successfully!")
    print("="*70 + "\n")


def load_and_use_optimized_parameters():
    """
    Example of loading previously optimized parameters and using them.
    """
    print("\n" + "="*70)
    print("Loading and Using Optimized Parameters")
    print("="*70 + "\n")

    param_file = "output/optimized_parameters.pkl"

    if not os.path.exists(param_file):
        print(f"No saved parameters found at {param_file}")
        print("Run the optimization example first.")
        return

    # Load parameters
    optimizer = ParameterOptimizer()
    optimizer.load_parameters(param_file)

    print("Loaded optimized parameters:")
    summary = optimizer.get_parameter_summary()
    for param_name in ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']:
        print(f"  {param_name:8s}: mean={summary[param_name]['mean']:10.6f}")
    print()

    # Use parameters in NoahPy simulation
    forcing_file = os.path.abspath("data/forcing.txt")
    if os.path.exists(forcing_file):
        print("Running NoahPy with optimized parameters...")
        Date, STC, SH2O = noah_main(
            forcing_file,
            trained_parameter=optimizer.parameters,
            lstm_model=None,
            output_flag=True  # Will save output to CSV
        )
        print(f"Simulation complete! Output shape: STC={STC.shape}, SH2O={SH2O.shape}")

    print()


if __name__ == "__main__":
    # Run the complete optimization example
    run_optimization_example()

    # Demonstrate loading and using saved parameters
    # load_and_use_optimized_parameters()
