@echo off
echo Downgrading Cellpose to v3.x for traditional model comparison...
echo.

echo Step 1: Uninstalling current Cellpose version...
pip uninstall cellpose -y

echo.
echo Step 2: Installing Cellpose v3.0.10 (last stable v3.x release)...
pip install "cellpose==3.0.10"

echo.
echo Step 3: Verifying installation...
python -c "import cellpose; print(f'Cellpose version: {cellpose.__version__}')"

echo.
echo Step 4: Testing model loading...
python -c "from cellpose import models; print('Testing model types:'); models.CellposeModel(model_type='nuclei'); print('nuclei - OK'); models.CellposeModel(model_type='cyto2'); print('cyto2 - OK'); models.CellposeModel(model_type='cyto3'); print('cyto3 - OK')"

echo.
echo Cellpose downgrade completed!
echo You can now run meaningful model comparisons.
pause 