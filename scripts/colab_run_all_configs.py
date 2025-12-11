# Google Colab Code Block - Run All Config Files
# Copy this entire block into a Google Colab cell

python_file_to_run = "src.segmentation_pipeline"

import glob
import os
import subprocess
import sys

# Get all config files from config/xenium_HE (excluding comparison files)
config_files = sorted(glob.glob("config/xenium_HE/processing_config_*.json"))
config_files = [f for f in config_files if "comparison" not in os.path.basename(f)]

print(f"Found {len(config_files)} configuration files to process\n")
print("=" * 80)

# Process each config file
try:
    for idx, config_file in enumerate(config_files, 1):
        config_name = os.path.basename(config_file)
        print(f"\n[{idx}/{len(config_files)}] Processing: {config_name}")
        print("-" * 80)
        
        # Run the segmentation pipeline for this config file
        # Use subprocess.Popen() for real-time streaming of output
        process = subprocess.Popen(
            [sys.executable, "-m", python_file_to_run, "--config", config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
        except KeyboardInterrupt:
            print("\n\n⚠ Execution interrupted by user (Ctrl+C)")
            process.terminate()
            process.wait()
            print("Stopping processing...")
            raise
        
        # Wait for process to complete and get return code
        returncode = process.wait()
        
        if returncode != 0:
            print(f"\n⚠ Warning: {config_name} exited with code {returncode}")
        
        print("-" * 80)

except KeyboardInterrupt:
    print("\n\n⚠ Execution interrupted by user (Ctrl+C)")
    print("Stopping processing...")
    sys.exit(1)

print(f"\n✓ Completed processing {len(config_files)} configuration files")

