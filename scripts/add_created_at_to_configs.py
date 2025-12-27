#!/usr/bin/env python3
"""Utility script to add created_at timestamps to config files in optimization project JSON files.

Usage:
    python scripts/add_created_at_to_configs.py <project_file.opt.json>
    
Or from project root:
    python -m scripts.add_created_at_to_configs <project_file.opt.json>
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

def add_created_at_to_configs(project_filepath: str, dry_run: bool = False):
    """Add created_at timestamps to config files that don't have them.
    
    Args:
        project_filepath: Path to the optimization project JSON file
        dry_run: If True, only print what would be changed without saving
    """
    if not os.path.exists(project_filepath):
        print(f"Error: Project file not found: {project_filepath}")
        return False
    
    # Load the project file
    with open(project_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get project created_at as fallback
    project_created_at = data.get('created_at')
    if isinstance(project_created_at, str):
        fallback_time = project_created_at
    else:
        fallback_time = datetime.now().isoformat()
    
    # Check and update config files
    updated_count = 0
    if 'config_files' in data:
        for i, cf_data in enumerate(data['config_files']):
            if 'created_at' not in cf_data:
                cf_data['created_at'] = fallback_time
                updated_count += 1
                print(f"  Config {i+1}: {cf_data.get('filepath', 'unknown')}")
                print(f"    Added created_at: {fallback_time}")
    
    if updated_count == 0:
        print("No config files needed updating (all already have created_at).")
        return True
    
    if dry_run:
        print(f"\n[DRY RUN] Would update {updated_count} config file(s).")
        print("Run without --dry-run to save changes.")
        return True
    
    # Save the updated file
    try:
        # Create backup
        backup_path = project_filepath + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nBackup saved to: {backup_path}")
        
        # Save updated file
        with open(project_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Successfully updated {updated_count} config file(s) in: {project_filepath}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/add_created_at_to_configs.py <project_file.opt.json> [--dry-run]")
        sys.exit(1)
    
    project_filepath = sys.argv[1]
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    
    print(f"Processing: {project_filepath}")
    if dry_run:
        print("[DRY RUN MODE - no changes will be saved]")
    print()
    
    success = add_created_at_to_configs(project_filepath, dry_run=dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()


