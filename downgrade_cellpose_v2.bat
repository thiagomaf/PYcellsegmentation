@echo off
echo Downgrading Cellpose to v3.x for traditional model comparison...
echo.

echo Step 1: Installing compatible numpy first...
pip install "numpy>=1.20.0,<2.0.0" --only-binary=all

echo.
echo Step 2: Installing Cellpose v3.0.10...
pip install "cellpose==3.0.10" --only-binary=all

echo.
echo If that fails, trying with pre-release and alternative sources...
pip install "cellpose==3.0.10" --pre --find-links https://download.pytorch.org/whl/torch_stable.html

echo.
echo Step 3: Verifying installation...
python -c "import cellpose; print(f'Cellpose version: {cellpose.__version__}')"

echo.
echo Cellpose downgrade completed!
pause 