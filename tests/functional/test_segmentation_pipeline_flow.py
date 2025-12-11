import pytest
import json
import os
import logging
import tifffile
import numpy as np
from click.testing import CliRunner # Can be useful for testing CLI scripts, or use subprocess

# It's often better to import the main function of the script to test it programmatically
from src.segmentation_pipeline import main as pipeline_main 
from src.file_paths import PROJECT_ROOT

# Minimal parameter_sets.json for functional test
FUNCTIONAL_TEST_CONFIG = {
    "image_configurations": [
        {
            "image_id": "func_test_img1",
            "original_image_filename": "images/func_dummy_img.tif",
            "is_active": True,
            "segmentation_options": {
                "apply_segmentation": True,
                # No rescaling or tiling for this basic functional test to keep it simple
            }
        }
    ],
    "cellpose_parameter_configurations": [
        {
            "param_set_id": "func_test_pset1",
            "is_active": True,
            "cellpose_parameters": {"MODEL_CHOICE": "cyto2", "DIAMETER": 0}
        }
    ],
    "global_segmentation_settings": {
        "MAX_PARALLEL_PROCESSES": 1,
        "default_log_level": "DEBUG" # Use DEBUG for tests to get more output if needed
    }
}

@pytest.fixture
def functional_test_environment(tmp_path, mocker):
    """Sets up a temporary environment for a functional pipeline run."""
    # Create temporary directories for images and results within tmp_path
    test_project_root = tmp_path
    images_dir = test_project_root / "images"
    images_dir.mkdir()
    results_dir = test_project_root / "results"
    results_dir.mkdir()
    # Cache and tiled output dirs (pipeline might try to create them)
    (test_project_root / "images" / "rescaled_cache").mkdir(exist_ok=True)
    (test_project_root / "images" / "tiled_outputs").mkdir(exist_ok=True)

    # Create a dummy image file
    dummy_img_array = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
    dummy_image_path = images_dir / "func_dummy_img.tif"
    tifffile.imwrite(dummy_image_path, dummy_img_array)

    # Create the config file for this test run
    config_file_path = test_project_root / "functional_test_params.json"
    with open(config_file_path, 'w') as f:
        json.dump(FUNCTIONAL_TEST_CONFIG, f)

    # Mock file_paths constants to point to our temporary directories
    mocker.patch('src.segmentation_pipeline.PROJECT_ROOT', str(test_project_root))
    mocker.patch('src.segmentation_pipeline.IMAGE_DIR_BASE', str(images_dir))
    mocker.patch('src.segmentation_pipeline.RESULTS_DIR_BASE', str(results_dir))
    mocker.patch('src.segmentation_pipeline.RESCALED_IMAGE_CACHE_DIR', str(images_dir / "rescaled_cache"))
    mocker.patch('src.segmentation_pipeline.TILED_IMAGE_OUTPUT_BASE', str(images_dir / "tiled_outputs"))
    
    # Also need to mock for pipeline_config_parser if it's not using the same instances
    mocker.patch('src.pipeline_config_parser.PROJECT_ROOT', str(test_project_root))
    mocker.patch('src.pipeline_config_parser.IMAGE_DIR_BASE', str(images_dir))
    # mocker.patch('src.pipeline_config_parser.RESULTS_DIR_BASE', str(results_dir)) # This was causing an error
    mocker.patch('src.pipeline_config_parser.RESCALED_IMAGE_CACHE_DIR', str(images_dir / "rescaled_cache"))
    mocker.patch('src.pipeline_config_parser.TILED_IMAGE_OUTPUT_BASE', str(images_dir / "tiled_outputs"))

    return str(config_file_path), results_dir, FUNCTIONAL_TEST_CONFIG # Return config dict for modification

def test_segmentation_pipeline_functional_run_success(functional_test_environment, mocker, caplog):
    """
    Tests a basic functional run of the segmentation pipeline with a successful worker.
    """
    caplog.set_level(logging.DEBUG)
    config_path, results_dir, _ = functional_test_environment # Original config is fine

    mock_worker_result = {
        "status": "succeeded", 
        "experiment_id_final": "func_test_img1_func_test_pset1", 
        "processing_unit_name": "func_dummy_img.tif", 
        "num_cells": 10,
        "mask_path": str(results_dir / "func_test_img1_func_test_pset1" / "func_dummy_img_mask.tif"),
        "message": "Mocked success"
    }
    mocked_worker = mocker.patch('src.segmentation_pipeline.segment_image_worker', return_value=mock_worker_result)

    test_args = ["prog_name", "--config", config_path, "--log_level", "DEBUG"]
    mocker.patch('sys.argv', test_args)

    try:
        pipeline_main()
    except SystemExit as e:
        assert e.code == 0, f"Pipeline exited with code {e.code} during success test"

    mocked_worker.assert_called_once()
    call_args = mocked_worker.call_args[0][0]
    assert call_args["experiment_id_final"] == "func_test_img1_func_test_pset1"

    run_log_path = results_dir / "run_log.json"
    assert run_log_path.exists()
    with open(run_log_path, 'r') as f:
        run_log_data = json.load(f)
    assert len(run_log_data) == 1
    assert run_log_data[0]["status"] == "succeeded"
    assert f"Job 1/1: func_test_img1_func_test_pset1 (Unit: func_dummy_img.tif) - SUCCEEDED" in caplog.text

def test_segmentation_pipeline_functional_run_worker_failure(functional_test_environment, mocker, caplog):
    """
    Tests a functional run where the segmentation worker reports a failure.
    """
    caplog.set_level(logging.DEBUG)
    config_path, results_dir, _ = functional_test_environment

    # Mock the segment_image_worker to return a failure
    mock_worker_failure_result = {
        "status": "failed", 
        "experiment_id_final": "func_test_img1_func_test_pset1", 
        "processing_unit_name": "func_dummy_img.tif", 
        "message": "Mocked worker internal error",
        "error_details": "Something went very wrong in the mock."
    }
    mocked_worker = mocker.patch('src.segmentation_pipeline.segment_image_worker', return_value=mock_worker_failure_result)

    test_args = ["prog_name", "--config", config_path, "--log_level", "DEBUG"]
    mocker.patch('sys.argv', test_args)

    try:
        pipeline_main()
    except SystemExit as e:
        assert e.code == 0, f"Pipeline exited with code {e.code} during failure test (expected normal exit)"

    # Assertions:
    # 1. Worker was called
    mocked_worker.assert_called_once()
    call_args = mocked_worker.call_args[0][0]
    assert call_args["experiment_id_final"] == "func_test_img1_func_test_pset1"

    # 2. Check for run_log.json and its content
    run_log_path = results_dir / "run_log.json"
    assert run_log_path.exists()
    with open(run_log_path, 'r') as f:
        run_log_data = json.load(f)
    assert len(run_log_data) == 1
    assert run_log_data[0]["status"] == "failed"
    assert run_log_data[0]["message"] == "Mocked worker internal error"

    # 3. Check log messages for failure reporting
    assert f"Job 1/1: func_test_img1_func_test_pset1 (Unit: func_dummy_img.tif) - FAILED (Mocked worker internal error)" in caplog.text
    assert "Check individual experiment folders and 'error_log.txt' files for details on failures." in caplog.text 