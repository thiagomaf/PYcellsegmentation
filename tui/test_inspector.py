"""Test script to verify inspector functionality."""
import sys
from pathlib import Path

# Add project root to path
tui_dir = Path(__file__).parent
project_root = tui_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_inspector_import():
    """Test if we can import the Inspector."""
    print("Testing Inspector imports...")
    
    # Try different import paths
    import_paths = [
        'textual_dev.inspector',
        'textual_dev',
        'textual.inspector',
    ]
    
    for import_path in import_paths:
        try:
            module = __import__(import_path, fromlist=['Inspector'])
            Inspector = getattr(module, 'Inspector', None)
            if Inspector:
                print(f"✓ Successfully imported Inspector from {import_path}")
                print(f"  Inspector class: {Inspector}")
                return True
        except ImportError as e:
            print(f"✗ Failed to import from {import_path}: {e}")
        except Exception as e:
            print(f"✗ Error with {import_path}: {e}")
    
    print("\n✗ Could not import Inspector from any path")
    print("  Make sure textual-dev is installed: pip install textual-dev")
    return False

if __name__ == "__main__":
    success = test_inspector_import()
    sys.exit(0 if success else 1)

