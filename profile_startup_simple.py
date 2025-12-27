"""Simple command-line profiling of TUI app startup using cProfile module."""
import sys
from pathlib import Path

# This script is meant to be run with: python -m cProfile -o startup_profile.stats profile_startup_simple.py
# Or: python -m cProfile -s cumulative profile_startup_simple.py

if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import and create the app (this is where startup happens)
    from tui.app import PyCellSegTUI
    
    # Create app instance (this triggers __init__ and imports)
    # We don't call app.run() - we only want to profile startup
    app = PyCellSegTUI()
    
    print("App created successfully. Profiling complete.")





