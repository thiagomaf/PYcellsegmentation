# diagnose_cellpose.ps1

# --- Configuration ---
$VenvName = ".venv-cellpose-diag"
$ProjectDir = $PSScriptRoot # Assumes the script is in your project root, adjust if needed
$VenvPath = Join-Path -Path $ProjectDir -ChildPath $VenvName
$VenvPythonPath = Join-Path -Path $VenvPath -ChildPath "Scripts\python.exe"
$VenvPipPath = Join-Path -Path $VenvPath -ChildPath "Scripts\pip.exe"

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

# 3. Upgrade pip within the virtual environment
Write-Host "Upgrading pip in the virtual environment..."
& $VenvPipPath install --upgrade pip
if (-not $?) {
    Write-Error "Failed to upgrade pip."
    exit 1
}

# 4. Install cellpose[gui] in the virtual environment
Write-Host "Installing cellpose[gui] (this may take a while)..."
& $VenvPipPath install "cellpose[gui]" --no-cache-dir
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
try:
    import cellpose
    from cellpose import models
    from cellpose import io # Also import io for a fuller test

    print(f"Cellpose version: {cellpose.__version__}")
    print(f"Path to cellpose module: {cellpose.__file__}")
    print(f"Path to models module: {models.__file__}")
    print("-" * 50)
    print("Contents of cellpose.models module (attributes):")
    model_attributes = [attr for attr in dir(models) if not attr.startswith('_')]
    if not model_attributes:
        print("  (No public attributes found or models module empty?)")
    else:
        for attr in model_attributes:
            print(f"  - {attr}")
    print("-" * 50)
    
    has_cellpose_class = hasattr(models, 'Cellpose')
    print(f"Does cellpose.models have 'Cellpose' class? {has_cellpose_class}")

    if has_cellpose_class:
        print("Attempting to initialize Cellpose model (cyto)...")
        try:
            model = models.Cellpose(gpu=False, model_type='cyto')
            print("Cellpose model (cyto) initialized successfully.")
        except Exception as e_model:
            print(f"Error initializing Cellpose model: {e_model}")
    else:
        print("Skipping model initialization because 'Cellpose' class was not found in models.")

    print("-" * 50)
    has_imread_func = hasattr(io, 'imread')
    print(f"Does cellpose.io have 'imread' function? {has_imread_func}")
    if has_imread_func:
        print("cellpose.io.imread found.")
    else:
        print("cellpose.io.imread NOT found.")

except Exception as e:
    print(f"An error occurred during the import or diagnosis: {e}")
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
