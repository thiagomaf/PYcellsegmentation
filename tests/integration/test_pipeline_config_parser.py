import pytest
import json
import os
from src.pipeline_config_parser import load_and_expand_configurations
from src.file_paths import PROJECT_ROOT # Assuming this is correctly set up

# Minimal valid parameter_sets.json content for testing
SAMPLE_CONFIG_CONTENT = {
    "image_configurations": [
        {
            "image_id": "test_img1",
            "original_image_filename": "dummy_image1.tif",
            "is_active": True,
            "mpp_x": 0.1, "mpp_y": 0.1,
            "segmentation_options": {
                "apply_segmentation": True,
                "rescaling_config": {"scale_factor": 0.5},
                "tiling_parameters": {"apply_tiling": False} # No tiling for this simple test
            }
        }
    ],
    "cellpose_parameter_configurations": [
        {
            "param_set_id": "test_pset1",
            "is_active": True,
            "cellpose_parameters": {
                "MODEL_CHOICE": "cyto3",
                "DIAMETER": 30
            }
        }
    ],
    "global_segmentation_settings": {
        "MAX_PARALLEL_PROCESSES": 1,
        "default_log_level": "INFO"
    }
}

@pytest.fixture
def create_dummy_image_files(tmp_path):
    # Create dummy image directory and files needed by the config parser
    # This simulates the structure expected by pipeline_config_parser
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    dummy_image_path = images_dir / "dummy_image1.tif"
    dummy_image_path.write_text("dummy image content") # Not a real tiff, but file existence is checked
    return images_dir

def test_load_and_expand_configurations_simple(tmp_path, mocker, create_dummy_image_files):
    """Test basic job list generation from a simple config."""
    # Mock rescale_image_and_save to avoid actual image processing and filesystem writes beyond test_data
    # It should return the (potentially new) path and the scale factor
    # For this test, assume rescale_image_and_save creates a uniquely named file if it rescales
    # or returns original path if scale is 1.0 or no rescale.
    # The important part is that load_and_expand_configurations receives a plausible path back.
    
    # Path to the rescaled image that the mock will return
    # It needs to be relative to where the test runs or an absolute path within tmp_path
    # Since rescale_image_and_save in pipeline_utils uses RESCALED_IMAGE_CACHE_DIR, 
    # we might need to mock that path or ensure the mock returns a path that doesn't cause issues.
    
    # For simplicity, let's make the mock return a path that suggests rescaling happened
    # and assume the cache directory logic within rescale_image_and_save is tested elsewhere or trusted.
    images_dir_abs = create_dummy_image_files # This is tmp_path / "images"
    
    # Define what the mocked rescale_image_and_save should return
    # (path_to_rescaled_image, applied_scale_factor)
    # The path should be something load_and_expand_configurations can use.
    # Let's make it put a dummy rescaled file in a subdir of tmp_path to simulate cache.
    mock_rescaled_cache_dir = tmp_path / "rescaled_cache" / "test_img1"
    mock_rescaled_cache_dir.mkdir(parents=True, exist_ok=True)
    mock_rescaled_path = mock_rescaled_cache_dir / "dummy_image1_scaled_0_5.tif"
    mock_rescaled_path.write_text("dummy rescaled content")
    
    mocker.patch('src.pipeline_config_parser.rescale_image_and_save',
                   return_value=(str(mock_rescaled_path), 0.5))
    
    # Mock tile_image as it's not being tested here and involves filesystem ops
    mocker.patch('src.pipeline_config_parser.tile_image', return_value=None) # No tiles generated

    # Setup paths to use tmp_path for this test
    mocker.patch('src.pipeline_config_parser.IMAGE_DIR_BASE', str(images_dir_abs))
    mocker.patch('src.pipeline_config_parser.RESCALED_IMAGE_CACHE_DIR', str(tmp_path / "rescaled_cache"))
    mocker.patch('src.pipeline_config_parser.TILED_IMAGE_OUTPUT_BASE', str(tmp_path / "tiled_outputs"))

    param_json_path = tmp_path / "test_params.json"
    with open(param_json_path, 'w') as f:
        json.dump(SAMPLE_CONFIG_CONTENT, f)

    jobs = load_and_expand_configurations(str(param_json_path))

    assert len(jobs) == 1
    job = jobs[0]
    assert job["original_image_id_for_log"] == "test_img1"
    assert job["param_set_id_for_log"] == "test_pset1"
    assert job["scale_factor_applied_for_log"] == 0.5
    assert job["MODEL_CHOICE"] == "cyto3"
    assert job["DIAMETER_FOR_CELLPOSE"] == 15 # 30 * 0.5
    assert job["processing_unit_name"] == "dummy_image1_scaled_0_5.tif"
    assert job["actual_image_path_to_process"] == str(mock_rescaled_path)
    assert job["experiment_id_final"] == "test_img1_test_pset1_scaled_0_5"
    assert job["mpp_x_original_for_log"] == 0.1 