# NoahPy Parameter Optimization Examples

This directory contains examples demonstrating how to use the parameter optimization module for NoahPy.

## Overview

NoahPy includes a gradient-based parameter optimization framework that allows calibrating soil parameters using observed data. The optimization module supports:

- **Multiple optimizers**: Adam, AdamW, SGD
- **Various loss functions**: MSE, MAE, RMSE, NSE (Nash-Sutcliffe Efficiency)
- **Physical constraints**: Ensures parameters stay within physically realistic bounds
- **Automatic differentiation**: Built on PyTorch for efficient gradient computation
- **Training utilities**: Learning rate scheduling, gradient clipping, early stopping

## Quick Start

### Basic Usage

```python
from NoahPy.optimization import ParameterOptimizer
from NoahPy.NoahPy import noah_main

# Initialize optimizer
optimizer = ParameterOptimizer(
    n_layers=10,           # Number of soil layers to optimize
    optimizer_type='Adam',
    lr=0.001,
    device='cpu'
)

# Initialize parameters (from soil type table)
parameters = optimizer.initialize_parameters(soil_type_index=7)

# Define model wrapper
def model_wrapper(params, **kwargs):
    Date, STC, SH2O = noah_main(
        "data/forcing.txt",
        trained_parameter=params,
        output_flag=False
    )
    return STC[:, 0]  # Return first layer temperature

# Run optimization
history = optimizer.train(
    model_fn=model_wrapper,
    train_data=observed_temperatures,
    n_epochs=100,
    loss_type='mse'
)

# Save optimized parameters
optimizer.save_parameters("optimized_params.pkl")
```

## Examples

### parameter_optimization_example.py

Complete end-to-end example showing:

1. **Data preparation**: Loading forcing data and creating observed data
2. **Optimizer setup**: Initializing parameters and optimizer
3. **Training loop**: Running optimization with validation
4. **Parameter analysis**: Examining parameter changes
5. **Visualization**: Plotting results and diagnostics
6. **Persistence**: Saving and loading parameters

Run the example:

```bash
python examples/parameter_optimization_example.py
```

This will:
- Generate synthetic observed data
- Optimize soil parameters for top 10 layers
- Save results to `output/` directory
- Create visualization plots

## Optimizable Parameters

The optimization framework focuses on 4 key soil hydraulic parameters:

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| BEXP | Pore size distribution index | 2.7 - 12.0 | - |
| SMCMAX | Saturated soil moisture content | 0.13 - 0.50 | m³/m³ |
| DKSAT | Saturated hydraulic conductivity | 1e-7 - 2e-4 | m/s |
| PSISAT | Saturated matric potential | 0.03 - 0.80 | m |

These parameters control soil water movement and heat transfer in the Noah LSM.

## Physical Constraints

The optimizer automatically enforces physical constraints to ensure parameters remain realistic:

```python
from NoahPy.optimization import PhysicalConstraints

# Check if parameters are valid
validation = PhysicalConstraints.validate_parameters(parameters)

# Apply constraints (clamp to valid range)
constrained_params = PhysicalConstraints.apply_constraints(parameters)
```

Constraints are based on the range of values in the SOILPARM.TBL lookup table.

## Advanced Features

### Custom Loss Functions

```python
def custom_loss(simulated, observed):
    # Example: Weighted MSE with emphasis on peak values
    weights = torch.where(observed > observed.median(), 2.0, 1.0)
    return optimizer.compute_loss(simulated, observed, weights=weights)
```

### Multi-Variable Optimization

```python
# Optimize for both soil temperature and moisture
def multi_objective_model(params, **kwargs):
    Date, STC, SH2O = noah_main(..., trained_parameter=params)
    return torch.cat([STC[:, 0], SH2O[:, 0]])  # Combine both outputs

# Use combined observed data
observed_combined = torch.cat([obs_temp, obs_moisture])
```

### Custom Optimizer Configuration

```python
from NoahPy.optimization import create_optimizer_from_config

config = {
    'n_layers': 10,
    'optimizer_type': 'AdamW',
    'lr': 0.005,
    'weight_decay': 1e-4,
    'clip_grad': 0.5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

optimizer = create_optimizer_from_config(config)
```

## Tips for Successful Optimization

1. **Start with good initial values**: Use parameters from the soil type that best matches your site

2. **Use validation data**: Always hold out data for validation to detect overfitting

3. **Monitor gradients**: The optimizer includes gradient clipping, but check for exploding/vanishing gradients

4. **Adjust learning rate**: Start with 0.001-0.01, reduce if loss oscillates

5. **Check physical validity**: Always validate that optimized parameters are physically reasonable

6. **Consider parameter correlation**: Some parameters are highly correlated (e.g., DKSAT and BEXP)

7. **Use appropriate loss function**: NSE is often better for hydrological applications than MSE

## Output Files

After running optimization, you'll find:

- `output/optimized_parameters.pkl`: Saved parameters (can be loaded and reused)
- `output/optimization_results.png`: Diagnostic plots
- `NoahPy_output.csv`: Model output with optimized parameters (if output_flag=True)

## Troubleshooting

**Problem**: Loss not decreasing

- Try reducing learning rate
- Check if gradients are flowing (print parameter.grad)
- Verify observed data is reasonable
- Increase number of epochs

**Problem**: Parameters hitting bounds

- Check if initial values are reasonable
- Consider if bounds are too restrictive
- Look for issues in observed data

**Problem**: Overfitting (train loss << val loss)

- Add weight decay (L2 regularization)
- Reduce number of optimizable layers
- Get more training data
- Use early stopping

## Related Documentation

- [Mathematical and Physical Principles](../docs/Mathematical_Physical_Principles_Summary.md)
- [NoahPy Technical Notes](../docs/NoahPy%20tech%20notes%20CHS-%20v1.1.pdf)
- [Test Suite](../tests/test_optimization.py)

## Citation

If you use the parameter optimization framework in your research, please cite:

```
NoahPy: A differentiable implementation of the Noah Land Surface Model
with gradient-based parameter optimization capabilities.
```
