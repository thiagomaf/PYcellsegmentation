@echo off
echo Using conda to install Cellpose v3.x for traditional model comparison...
echo.

echo Step 1: Checking conda availability...
conda --version

echo.
echo Step 2: Installing Cellpose v3.0.10 via conda-forge...
conda install -c conda-forge "cellpose=3.0.10" -y

echo.
echo Step 3: If conda fails, trying pip with specific numpy version...
pip install "numpy==1.24.3" --force-reinstall
pip install "cellpose==3.0.10" --no-deps

echo.
echo Step 4: Installing remaining dependencies...
pip install torch torchvision tifffile opencv-python-headless matplotlib scipy scikit-image

echo.
echo Step 5: Verifying installation...
python -c "import cellpose; print(f'Cellpose version: {cellpose.__version__}')"

echo.
echo Step 6: Testing model loading...
python -c "from cellpose import models; print('Testing model types:'); m1=models.CellposeModel(model_type='nuclei'); print('nuclei - OK'); m2=models.CellposeModel(model_type='cyto2'); print('cyto2 - OK'); m3=models.CellposeModel(model_type='cyto3'); print('cyto3 - OK')"

echo.
echo Cellpose downgrade completed!
pause 