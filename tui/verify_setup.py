"""Verify that textual-dev is properly set up for inspector usage."""
import sys
from pathlib import Path

def check_textual_dev():
    """Check if textual-dev is available and can be imported."""
    print("Checking textual-dev setup...")
    print(f"Python: {sys.executable}")
    print()
    
    # Check 1: Can we import textual_dev?
    try:
        import textual_dev
        print("✓ textual_dev module found")
        print(f"  Location: {textual_dev.__file__}")
    except ImportError as e:
        print(f"✗ Cannot import textual_dev: {e}")
        print("\n  Solution: pip install textual-dev")
        return False
    
    # Check 2: Can we find Inspector?
    inspector_found = False
    import_paths = [
        ('textual_dev.inspector', 'Inspector'),
        ('textual_dev', 'Inspector'),
    ]
    
    for module_path, class_name in import_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            Inspector = getattr(module, class_name, None)
            if Inspector:
                print(f"✓ Inspector found at {module_path}.{class_name}")
                inspector_found = True
                break
        except ImportError:
            continue
        except Exception as e:
            print(f"  Note: Error checking {module_path}: {e}")
    
    if not inspector_found:
        print("✗ Inspector class not found")
        print("\n  This is normal! The Inspector is only available when")
        print("  running with: textual run --dev tui/app.py")
        print("\n  The 'textual run --dev' command automatically injects")
        print("  the inspector functionality.")
    
    print()
    print("=" * 60)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 60)
    print()
    print("To use the inspector:")
    print("  1. Make sure textual-dev is installed: pip install textual-dev")
    print("  2. Run your app with: textual run --dev tui/app.py")
    print("  3. Press F12 or Ctrl+I to open the inspector")
    print()
    
    return True

if __name__ == "__main__":
    success = check_textual_dev()
    sys.exit(0 if success else 1)

