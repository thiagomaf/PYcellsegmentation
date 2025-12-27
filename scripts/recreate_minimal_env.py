#!/usr/bin/env python3
"""Script to wipe and recreate .pycellseg environment with minimal dependencies."""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and print the result."""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {description} failed")
        if result.stderr:
            print(f"Error output: {result.stderr}")
        if check:
            sys.exit(1)
    if result.stdout:
        print(result.stdout)
    return result.returncode == 0

def main():
    """Main setup function."""
    project_root = Path(__file__).parent.parent
    venv_path = project_root / ".pycellseg"
    
    print("="*60)
    print("Recreating .pycellseg environment with minimal dependencies")
    print("="*60)
    
    # Step 1: Remove existing environment
    if venv_path.exists():
        print(f"\nRemoving existing environment at: {venv_path}")
        print("Note: On Windows, you may need to close any Python processes first.")
        try:
            # Try to remove with retries for Windows file locking issues
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(venv_path)
                    print("✓ Environment removed")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"  Retry {attempt + 1}/{max_retries}... (files may be locked)")
                        time.sleep(1)
                    else:
                        print(f"\n⚠ Could not remove environment: {e}")
                        print("Please manually delete .pycellseg folder and run this script again.")
                        print("Or close any Python processes using the environment and retry.")
                        sys.exit(1)
        except Exception as e:
            print(f"\n⚠ Error removing environment: {e}")
            print("Please manually delete .pycellseg folder and run this script again.")
            sys.exit(1)
    else:
        print("\nNo existing environment found, creating new one")
    
    # Step 2: Create new virtual environment
    print(f"\nCreating new virtual environment...")
    python_exe = sys.executable
    run_command(
        f'"{python_exe}" -m venv .pycellseg',
        "Creating virtual environment"
    )
    
    # Determine the correct python executable path
    if os.name == 'nt':  # Windows
        venv_python = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/Mac
        venv_python = venv_path / "bin" / "python"
    
    if not venv_python.exists():
        print(f"Error: Virtual environment python not found at {venv_python}")
        sys.exit(1)
    
    # Step 3: Upgrade pip
    run_command(
        f'"{venv_python}" -m pip install --upgrade pip',
        "Upgrading pip"
    )
    
    # Step 4: Install minimal core packages
    core_packages = [
        "textual>=0.60.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
    ]
    
    print("\n" + "="*60)
    print("Installing core packages (minimal set)...")
    print("="*60)
    
    for package in core_packages:
        run_command(
            f'"{venv_python}" -m pip install "{package}"',
            f"Installing {package}"
        )
    
    # Step 5: Verify installation
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    
    run_command(
        f'"{venv_python}" -m pip list',
        "Listing installed packages"
    )
    
    print("\n" + "="*60)
    print("✓ Minimal environment setup complete!")
    print("="*60)
    print("\nCore packages installed:")
    print("  - textual (TUI framework)")
    print("  - pydantic (data models)")
    print("  - rich (text formatting)")
    print("\nOptional packages (will be lazy-loaded when needed):")
    print("  - numpy, tifffile (for stats calculation)")
    print("  - plotext (for visualization)")
    print("  - scikit-learn, pyDOE3 (for advanced visualization)")
    print("\nTo use the TUI, activate the environment:")
    if os.name == 'nt':
        print("  .pycellseg\\Scripts\\Activate.ps1")
    else:
        print("  source .pycellseg/bin/activate")
    print("="*60)

if __name__ == '__main__':
    main()

