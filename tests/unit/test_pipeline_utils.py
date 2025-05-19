import pytest
from src.pipeline_utils import construct_full_experiment_id, get_base_experiment_id, format_scale_factor_for_path, clean_filename_for_dir

# Test cases for construct_full_experiment_id
@pytest.mark.parametrize(
    "image_id, param_set_id, scale_factor, processing_unit_name_for_tile, is_tile, expected_id",
    [
        ("img1", "pset1", None, None, False, "img1_pset1"),
        ("img1", "pset1", 1.0, None, False, "img1_pset1"),
        ("img1", "pset1", 0.5, None, False, "img1_pset1_scaled_0_5"),
        ("img2", "pset_long_name", 0.25, None, False, "img2_pset_long_name_scaled_0_25"),
        ("imgT", "psetT", 1.0, "tile_r1_c2.tif", True, "imgT_psetT_tile_r1_c2"),
        ("imgS", "psetS", 0.5, "tile_scaled_r0_c0.tif", True, "imgS_psetS_tile_scaled_r0_c0"),
        # Test with unusual characters in tile name (expecting clean_filename_for_dir to handle it)
        ("imgX", "psetX", None, "tile with spaces & chars!.tif", True, "imgX_psetX_tile_with_spaces___chars_"),
    ]
)
def test_construct_full_experiment_id(image_id, param_set_id, scale_factor, processing_unit_name_for_tile, is_tile, expected_id):
    assert construct_full_experiment_id(image_id, param_set_id, scale_factor, processing_unit_name_for_tile, is_tile) == expected_id

@pytest.mark.parametrize(
    "filename, expected",
    [
        ("my_image.tif", "my_image"),
        ("my image with spaces.ome.tiff", "my_image_with_spaces_ome"),
        ("complex-name.v1.2.ext", "complex-name_v1_2"),
        ("no_extension_here", "no_extension_here"),
        ("file.with.dots.ext", "file_with_dots"),
        ("already_clean", "already_clean"),
        ("a!b@c#d$.tif", "a_b_c_d_")
    ]
)
def test_clean_filename_for_dir(filename, expected):
    assert clean_filename_for_dir(filename) == expected

@pytest.mark.parametrize(
    "scale_factor, expected_str",
    [
        (None, ""),
        (1.0, ""),
        (0.5, "_scaled_0_5"),
        (0.253, "_scaled_0_253"),
        (0.7, "_scaled_0_7"),
    ]
)
def test_format_scale_factor_for_path(scale_factor, expected_str):
    assert format_scale_factor_for_path(scale_factor) == expected_str 