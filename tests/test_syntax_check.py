"""
Syntax check for optimization module (no runtime dependencies)
"""

import ast
import sys
import os

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    """Check all optimization-related files."""
    files_to_check = [
        'NoahPy/optimization.py',
        'tests/test_optimization.py',
        'tests/test_optimization_simple.py',
        'examples/parameter_optimization_example.py',
    ]

    print("="*70)
    print("Syntax Check for Parameter Optimization Module")
    print("="*70)

    all_passed = True

    for filepath in files_to_check:
        full_path = os.path.join('/home/user/NoahPy', filepath)

        if not os.path.exists(full_path):
            print(f"\n✗ MISSING: {filepath}")
            all_passed = False
            continue

        valid, error = check_python_syntax(full_path)

        if valid:
            # Get file size and line count
            with open(full_path, 'r') as f:
                lines = f.readlines()
            size_kb = os.path.getsize(full_path) / 1024

            print(f"\n✓ VALID: {filepath}")
            print(f"  Lines: {len(lines)}, Size: {size_kb:.1f} KB")
        else:
            print(f"\n✗ SYNTAX ERROR: {filepath}")
            print(f"  Error: {error}")
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✓ All files passed syntax check!")
        print("="*70)
        return 0
    else:
        print("✗ Some files have issues")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
