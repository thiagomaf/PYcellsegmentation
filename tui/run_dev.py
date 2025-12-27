"""Run the TUI with developer tools enabled."""
import sys
from pathlib import Path

# Add the project root to the path
tui_dir = Path(__file__).parent
project_root = tui_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # This will be run via 'textual run --dev run_dev.py'
    # or you can use: python -m textual run --dev tui/run_dev.py
    from tui.app import PyCellSegTUI
    
    app = PyCellSegTUI()
    app.run()

