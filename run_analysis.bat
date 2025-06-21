@echo off
python src/analyze_cellpose_comparison.py --config config/processing_config_comparison5.json --results_dir results --output_dir results/comparison_analysis
echo.
echo Analysis completed. Check results/comparison_analysis/ for outputs.
pause 