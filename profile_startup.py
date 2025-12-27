"""Profile the TUI app startup to identify performance bottlenecks."""
import cProfile
import pstats
from pstats import SortKey
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def profile_app_startup():
    """Profile the TUI app startup."""
    # Create profiler
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    try:
        # Import and create the app (this is where startup happens)
        from tui.app import PyCellSegTUI
        
        # Create app instance (this triggers __init__ and imports)
        app = PyCellSegTUI()
        
        # Stop profiling before app.run() (we only want to profile startup)
        profiler.disable()
        
        # Save stats to file
        stats_file = project_root / "startup_profile.stats"
        profiler.dump_stats(str(stats_file))
        print(f"Profile data saved to: {stats_file}")
        
        # Also print a summary to console
        print("\n" + "="*80)
        print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
        print("="*80)
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(30)
        
        print("\n" + "="*80)
        print("TOP 30 FUNCTIONS BY TOTAL TIME")
        print("="*80)
        stats.sort_stats(SortKey.TIME)  # TIME is the total time spent in the function
        stats.print_stats(30)
        
        print("\n" + "="*80)
        print("FUNCTIONS IN tui/ DIRECTORY")
        print("="*80)
        stats.print_stats('tui/')
        
        print(f"\nTo analyze the full profile interactively, run:")
        print(f"  python -m pstats {stats_file}")
        print(f"\nThen in the pstats prompt, use commands like:")
        print(f"  sort cumulative")
        print(f"  stats 50")
        print(f"  stats tui/")
        
    except Exception as e:
        profiler.disable()
        print(f"Error during profiling: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    profile_app_startup()

