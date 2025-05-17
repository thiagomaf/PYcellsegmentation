# diagnose_cellpose.ps1

# --- Configuration ---
$VenvName = ".venv-cellpose-diag"
$ProjectDir = $PSScriptRoot # Assumes the script is in your project root, adjust if needed
$VenvPath = Join-Path -Path $ProjectDir -ChildPath $VenvName
$VenvPythonPath = Join-Path -Path $VenvPath -ChildPath "Scripts\python.exe"

# --- Script ---
Write-Host "Starting Cellpose Diagnostic Script..."
Write-Host "Virtual environment will be created at: $VenvPath"

# 1. Remove existing virtual environment if it exists
if (Test-Path $VenvPath) {
    Write-Host "Removing existing virtual environment: $VenvPath"
    Remove-Item -Recurse -Force $VenvPath -ErrorAction SilentlyContinue
}

# 2. Create a new Python virtual environment
Write-Host "Creating new virtual environment..."
python -m venv $VenvPath
if (-not $?) {
    Write-Error "Failed to create virtual environment. Ensure Python is installed and in PATH."
    exit 1
}

# 3. Upgrade pip within the virtual environment using python -m pip
Write-Host "Upgrading pip in the virtual environment..."
& $VenvPythonPath -m pip install --upgrade pip
if (-not $?) {
    Write-Host "Warning: Failed to upgrade pip. Continuing with existing pip version."
}

# 4. Install cellpose[gui] in the virtual environment
Write-Host "Installing cellpose[gui] (this may take a while)..."
& $VenvPythonPath -m pip install "cellpose[gui]" --no-cache-dir
if (-not $?) {
    Write-Error "Failed to install cellpose."
    exit 1
}

# 5. Set KMP_DUPLICATE_LIB_OK environment variable for this session
Write-Host "Setting KMP_DUPLICATE_LIB_OK=TRUE for this session."
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 6. Define the Python diagnostic script
$PythonScriptContent = @"
import os
print(f"KMP_DUPLICATE_LIB_OK is set to: {os.environ.get('KMP_DUPLICATE_LIB_OK')}")

print("Attempting to import cellpose and its modules...")
cellpose_module = None
models_module = None
io_module = None

try:
    import cellpose
    cellpose_module = cellpose
    print(f"Successfully imported 'cellpose' module.")
    
    if hasattr(cellpose_module, '__file__'):
        print(f"Path to cellpose module: {cellpose_module.__file__}")
    else:
        print("'cellpose' module has no __file__ attribute.")

    if hasattr(cellpose_module, '__version__'):
        print(f"Cellpose version: {cellpose_module.__version__}")
    else:
        print("'cellpose' module has no __version__ attribute.")

    try:
        from cellpose import models
        models_module = models
        print(f"Successfully imported 'cellpose.models' module.")
        if hasattr(models_module, '__file__'):
            print(f"Path to models module: {models_module.__file__}")
        else:
            print("'cellpose.models' module has no __file__ attribute.")
    except ImportError as e_models:
        print(f"Failed to import 'cellpose.models': {e_models}")
    
    try:
        from cellpose import io
        io_module = io
        print(f"Successfully imported 'cellpose.io' module.")
        if hasattr(io_module, '__file__'):
            print(f"Path to io module: {io_module.__file__}")
        else:
            print("'cellpose.io' module has no __file__ attribute.")
    except ImportError as e_io:
        print(f"Failed to import 'cellpose.io': {e_io}")

    if models_module:
        print("-" * 50)
        print("Contents of cellpose.models module (attributes):")
        model_attributes = [attr for attr in dir(models_module) if not attr.startswith('_')]
        if not model_attributes:
            print("  (No public attributes found or models module empty?)")
        else:
            for attr in model_attributes:
                print(f"  - {attr}")
        print("-" * 50)
        
        has_cellpose_class = hasattr(models_module, 'Cellpose')
        print(f"Does cellpose.models have 'Cellpose' class? {has_cellpose_class}")

        if has_cellpose_class:
            print("Attempting to initialize Cellpose model (cyto)...")
            try:
                model = models_module.Cellpose(gpu=False, model_type='cyto')
                print("Cellpose model (cyto) initialized successfully.")
            except Exception as e_model_init:
                print(f"Error initializing Cellpose model: {e_model_init}")
        else:
            print("Skipping model initialization because 'Cellpose' class was not found in models_module.")
    else:
        print("Skipping models_module diagnostics as it was not imported.")

    if io_module:
        print("-" * 50)
        has_imread_func = hasattr(io_module, 'imread')
        print(f"Does cellpose.io have 'imread' function? {has_imread_func}")
        if has_imread_func:
            print("cellpose.io.imread found.")
        else:
            print("cellpose.io.imread NOT found.")
    else:
        print("Skipping io_module diagnostics as it was not imported.")

except ImportError as e_cellpose:
    print(f"Failed to import 'cellpose' itself: {e_cellpose}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred during diagnosis: {e}")
    import traceback
    traceback.print_exc()

print("-" * 50)
print("Diagnostic script finished.")
"@

# 7. Write the Python script to a temporary file
$TempPythonScriptPath = Join-Path -Path $ProjectDir -ChildPath "_temp_diag.py"
Write-Host "Writing temporary Python diagnostic script to: $TempPythonScriptPath"
Set-Content -Path $TempPythonScriptPath -Value $PythonScriptContent -Encoding UTF8

# 8. Run the Python diagnostic script
Write-Host "Running Python diagnostic script using interpreter from: $VenvPythonPath"
Write-Host "--- Python Script Output Starts ---"
& $VenvPythonPath $TempPythonScriptPath
Write-Host "--- Python Script Output Ends ---"

# 9. Delete the temporary Python script
Write-Host "Deleting temporary Python diagnostic script: $TempPythonScriptPath"
Remove-Item $TempPythonScriptPath -ErrorAction SilentlyContinue

Write-Host "Diagnostic process complete. Please review the output above."
