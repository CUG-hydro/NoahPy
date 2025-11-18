"""
Code structure validation (no PyTorch required)
Tests that all classes, methods, and functions are properly defined
"""

import sys
import os
import importlib.util

def load_module_from_file(filepath, module_name):
    """Load a Python module from filepath."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        # Don't execute, just check if it can be compiled
        with open(filepath, 'r') as f:
            code = compile(f.read(), filepath, 'exec')
        return True
    except Exception as e:
        return str(e)

def check_class_methods(filepath, expected_classes):
    """Check if expected classes and methods exist."""
    with open(filepath, 'r') as f:
        content = f.read()

    results = {}
    for class_name, methods in expected_classes.items():
        class_found = f"class {class_name}" in content
        results[class_name] = {
            'found': class_found,
            'methods': {}
        }

        if class_found:
            for method in methods:
                method_found = f"def {method}" in content
                results[class_name]['methods'][method] = method_found

    return results

def main():
    print("\n" + "="*70)
    print("Code Structure Validation")
    print("="*70)

    # Test 1: Module compilation
    print("\n[Test 1] Module Compilation Check")
    print("-" * 70)

    optimization_path = '/home/user/NoahPy/NoahPy/optimization.py'
    result = load_module_from_file(optimization_path, 'optimization')

    if result is True:
        print("✓ optimization.py can be compiled")
    else:
        print(f"✗ Compilation error: {result}")
        return 1

    # Test 2: Class structure
    print("\n[Test 2] Class Structure Validation")
    print("-" * 70)

    expected_structure = {
        'PhysicalConstraints': [
            'apply_constraints',
            'validate_parameters',
        ],
        'ParameterOptimizer': [
            '__init__',
            'initialize_parameters',
            'compute_loss',
            'train_step',
            'validate',
            'train',
            'save_parameters',
            'load_parameters',
            'get_parameter_summary',
        ]
    }

    structure_results = check_class_methods(optimization_path, expected_structure)

    all_valid = True
    for class_name, info in structure_results.items():
        if info['found']:
            print(f"\n✓ Class '{class_name}' found")

            methods_valid = all(info['methods'].values())
            if methods_valid:
                print(f"  ✓ All {len(info['methods'])} methods present")
                for method, found in info['methods'].items():
                    print(f"    - {method}")
            else:
                print(f"  ✗ Missing methods:")
                for method, found in info['methods'].items():
                    if not found:
                        print(f"    ✗ {method}")
                        all_valid = False
        else:
            print(f"\n✗ Class '{class_name}' not found")
            all_valid = False

    # Test 3: Constants and bounds
    print("\n[Test 3] Constants and Bounds")
    print("-" * 70)

    with open(optimization_path, 'r') as f:
        content = f.read()

    bounds_check = "BOUNDS = {" in content
    if bounds_check:
        print("✓ BOUNDS dictionary defined")

        required_params = ['BEXP', 'SMCMAX', 'DKSAT', 'PSISAT']
        for param in required_params:
            if f"'{param}'" in content:
                print(f"  ✓ {param} bound defined")
            else:
                print(f"  ✗ {param} bound missing")
                all_valid = False
    else:
        print("✗ BOUNDS dictionary not found")
        all_valid = False

    # Test 4: Test files
    print("\n[Test 4] Test Files Structure")
    print("-" * 70)

    test_files = [
        'tests/test_optimization.py',
        'tests/test_optimization_simple.py',
    ]

    for test_file in test_files:
        full_path = os.path.join('/home/user/NoahPy', test_file)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                test_content = f.read()

            # Count test functions
            test_count = test_content.count('def test_')
            class_count = test_content.count('class Test')

            print(f"\n✓ {test_file}")
            print(f"  Test classes: {class_count}")
            print(f"  Test functions: {test_count}")
        else:
            print(f"\n✗ {test_file} not found")
            all_valid = False

    # Test 5: Example file
    print("\n[Test 5] Example File Structure")
    print("-" * 70)

    example_path = '/home/user/NoahPy/examples/parameter_optimization_example.py'
    if os.path.exists(example_path):
        with open(example_path, 'r') as f:
            example_content = f.read()

        required_functions = [
            'run_optimization_example',
            'prepare_observed_data',
            'load_and_use_optimized_parameters',
        ]

        print(f"✓ Example file found")
        for func in required_functions:
            if f"def {func}" in example_content:
                print(f"  ✓ Function '{func}' present")
            else:
                print(f"  ✗ Function '{func}' missing")
                all_valid = False
    else:
        print("✗ Example file not found")
        all_valid = False

    # Summary
    print("\n" + "="*70)
    if all_valid:
        print("✓ ALL STRUCTURE CHECKS PASSED")
        print("="*70)
        print("\nCode structure is correct. Waiting for PyTorch to run functional tests.")
        return 0
    else:
        print("✗ SOME STRUCTURE CHECKS FAILED")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
