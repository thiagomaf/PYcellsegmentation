#!/usr/bin/env python3
"""Script to set up a minimal TUI environment with only essential dependencies.

This removes heavy dependencies that slow down startup and keeps only what's needed
for the TUI to run quickly.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: {description} had exit code {result.returncode}")
        if result.stderr:
            print(f"Error: {result.stderr}")
    return result.returncode == 0

def main():
    """Main setup function."""
    venv_python = r".\.pycellseg\Scripts\python.exe"
    
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment not found at {venv_python}")
        print("Please create the .pycellseg environment first.")
        sys.exit(1)
    
    print("Setting up minimal TUI environment...")
    print("This will uninstall heavy dependencies and keep only essentials.")
    
    # Packages to uninstall (heavy dependencies that slow startup)
    heavy_packages = [
        "scikit-learn",
        "scipy", 
        "pyDOE3",
        "plotext",  # Optional - only needed for visualization
        "textual-plotext",  # Not needed - we use plotext directly
        "numpy",  # Will be lazy loaded only when needed for stats
        "joblib",
        "threadpoolctl",
    ]
    
    print("\nUninstalling heavy packages...")
    for package in heavy_packages:
        run_command(
            f'{venv_python} -m pip uninstall -y {package}',
            f"Uninstalling {package}"
        )
    
    # Core packages (minimal set)
    core_packages = [
        "textual>=0.60.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
    ]
    
    print("\nInstalling/upgrading core packages...")
    for package in core_packages:
        run_command(
            f'{venv_python} -m pip install --upgrade "{package}"',
            f"Installing {package}"
        )
    
    print("\n" + "="*60)
    print("Minimal TUI environment setup complete!")
    print("\nCore packages installed:")
    print("  - textual (TUI framework)")
    print("  - pydantic (data models)")
    print("  - rich (text formatting)")
    print("\nOptional packages (lazy loaded when needed):")
    print("  - numpy, tifffile (for stats calculation)")
    print("  - scikit-learn, pyDOE3, plotext (for visualization)")
    print("\nNote: textual-plotext is NOT needed - we use plotext directly.")
    print("These optional packages will be installed when their features are used.")
    print("="*60)

if __name__ == '__main__':
    main()

