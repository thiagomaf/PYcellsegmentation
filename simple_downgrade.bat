@echo off
echo Simple Cellpose v3.x installation...
echo.

echo Step 1: Installing pre-compiled numpy 1.24.3...
pip install "numpy==1.24.3"

echo.
echo Step 2: Installing Cellpose 3.0.10 without dependencies...
pip install "cellpose==3.0.10" --no-deps

echo.
echo Step 3: Installing other required dependencies...
pip install torch torchvision tifffile opencv-python-headless matplotlib scipy scikit-image roifile superqt natsort tqdm

echo.
echo Step 4: Verifying installation...
python -c "import cellpose; print(f'Cellpose version: {cellpose.__version__}')"

echo.
echo Done!
pause 