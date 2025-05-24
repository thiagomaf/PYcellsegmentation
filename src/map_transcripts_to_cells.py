import os
import argparse
import pandas as pd
import numpy as np
import tifffile # Or from cellpose import io
from scipy.sparse import coo_matrix
import scipy.io 
import traceback
from cellpose import io as cellpose_io # Using cellpose.io for consistency
from scipy.sparse import csr_matrix
from pyarrow.parquet import read_table
import logging
import json

# Internal imports from src
from .file_paths import PROJECT_ROOT, RESULTS_DIR_BASE # Assuming RESULTS_DIR_BASE is where segmentation outputs are
from .pipeline_utils import setup_logging, construct_full_experiment_id, construct_mask_path # Changed import

logger = logging.getLogger(__name__)

# --- Default Values ---
DEFAULT_QV_THRESHOLD = 20.0

def load_transcripts(transcript_file, qv_threshold):
    """Loads transcripts, likely from a Parquet file, filters by QV, and decodes feature_name."""
    print(f"Loading transcripts from: {transcript_file} ...")
    try:
        if transcript_file.endswith(".parquet"):
            df = pd.read_parquet(transcript_file)
        elif transcript_file.endswith(".csv"):
            df = pd.read_csv(transcript_file)
        else:
            raise ValueError("Unsupported transcript file format. Please use .parquet or .csv.")
        
        print(f"Loaded {len(df)} total transcripts.")
        
        required_cols = ['x_location', 'y_location', 'feature_name'] 
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in transcript data.")
        
        if 'qv' in df.columns:
            df_filtered = df[df['qv'] >= qv_threshold].copy()
            print(f"Filtered {len(df_filtered)} transcripts with QV >= {qv_threshold}.")
        else:
            print("Warning: 'qv' column not found in transcript data. No QV filtering applied.")
            df_filtered = df.copy()

        if df_filtered.empty:
            print("No transcripts remaining after QV filtering (or initial load was empty).")
            return df_filtered
            
        if 'feature_name' in df_filtered.columns and isinstance(df_filtered['feature_name'].iloc[0], bytes):
            print("Decoding 'feature_name' from bytes to string (UTF-8)...")
            df_filtered['feature_name'] = df_filtered['feature_name'].str.decode('utf-8')
        
        if 'transcript_id' not in df_filtered.columns:
             print("Warning: 'transcript_id' not found. Using index as transcript_id.")
             df_filtered['transcript_id'] = df_filtered.index.astype(str)
        else:
             df_filtered['transcript_id'] = df_filtered['transcript_id'].astype(str)

        return df_filtered
    except Exception as e:
        print(f"Error loading or filtering transcripts: {e}")
        traceback.print_exc()
        return None

def load_segmentation_mask(mask_file_path):
    print(f"Loading segmentation mask from: {mask_file_path} ...")
    try:
        mask = cellpose_io.imread(mask_file_path)
        print(f"Loaded mask with shape: {mask.shape}, dtype: {mask.dtype}, max_id: {np.max(mask)}")
        return mask
    except Exception as e:
        print(f"Error loading segmentation mask: {e}")
        traceback.print_exc()
        return None

def map_transcripts_to_cells(transcripts_df, mask_image, microns_per_pixel_x, microns_per_pixel_y, 
                               x_offset_microns=0, y_offset_microns=0):
    print("Mapping transcripts to cells...")
    if transcripts_df is None or transcripts_df.empty or mask_image is None:
        print("  Cannot map: transcript data or mask is missing or empty.")
        return transcripts_df 

    transcripts_to_map_df = transcripts_df.copy()

    transcripts_to_map_df['x_pixel'] = ((transcripts_to_map_df['x_location'] - x_offset_microns) / microns_per_pixel_x).astype(int)
    transcripts_to_map_df['y_pixel'] = ((transcripts_to_map_df['y_location'] - y_offset_microns) / microns_per_pixel_y).astype(int)
    
    mask_height, mask_width = mask_image.shape
    assigned_cell_ids = np.full(len(transcripts_to_map_df), -1, dtype=np.int32) 

    valid_indices_bool = (
        (transcripts_to_map_df['x_pixel'] >= 0) & (transcripts_to_map_df['x_pixel'] < mask_width) &
        (transcripts_to_map_df['y_pixel'] >= 0) & (transcripts_to_map_df['y_pixel'] < mask_height)
    )
    
    valid_ilocs = np.where(valid_indices_bool)[0]

    if valid_ilocs.size > 0:
        y_coords_for_lookup = transcripts_to_map_df['y_pixel'].iloc[valid_ilocs].values
        x_coords_for_lookup = transcripts_to_map_df['x_pixel'].iloc[valid_ilocs].values
        
        cell_ids_from_mask = mask_image[y_coords_for_lookup, x_coords_for_lookup]
        
        assigned_cell_ids[valid_ilocs] = np.where(cell_ids_from_mask > 0, cell_ids_from_mask, -1)

    transcripts_to_map_df['assigned_cell_id'] = assigned_cell_ids
    
    num_assigned = np.sum(transcripts_to_map_df['assigned_cell_id'] > 0)
    print(f"Assigned {num_assigned} transcripts to cells (out of {len(valid_ilocs)} within bounds).")
    return transcripts_to_map_df

def generate_feature_cell_matrix(mapped_transcripts_df, output_dir, filename_prefix="feature_cell_matrix"):
    print("Generating feature-cell matrix...")
    if mapped_transcripts_df is None or mapped_transcripts_df.empty:
        print("  Input DataFrame for feature-cell matrix is empty. Skipping matrix generation.")
        return

    assigned_df = mapped_transcripts_df[mapped_transcripts_df['assigned_cell_id'] > 0].copy()

    if assigned_df.empty:
        print("No transcripts were assigned to cells. Cannot generate feature-cell matrix.")
        return

    if isinstance(assigned_df['feature_name'].iloc[0], bytes):
        print("  Decoding 'feature_name' in generate_feature_cell_matrix (should have been done at load)...")
        assigned_df['feature_name'] = assigned_df['feature_name'].str.decode('utf-8')

    assigned_df['assigned_cell_id_str'] = assigned_df['assigned_cell_id'].astype(str) 
    
    genes = pd.Categorical(assigned_df['feature_name'])
    cells = pd.Categorical(assigned_df['assigned_cell_id_str']) 

    counts = coo_matrix((np.ones(len(assigned_df), dtype=np.int32), 
                         (genes.codes, cells.codes)), 
                        shape=(len(genes.categories), len(cells.categories)))

    matrix_path = os.path.join(output_dir, f"{filename_prefix}_matrix.mtx")
    features_path = os.path.join(output_dir, f"{filename_prefix}_features.tsv") 
    barcodes_path = os.path.join(output_dir, f"{filename_prefix}_barcodes.tsv")

    try:
        scipy.io.mmwrite(matrix_path, counts)
        print(f"Saved MTX matrix to: {matrix_path}")

        with open(features_path, 'w') as f:
            for gene_name in genes.categories:
                f.write(f"{gene_name}	{gene_name}	Gene Expression") 
        print(f"Saved features (genes) to: {features_path}")

        with open(barcodes_path, 'w') as f:
            for cell_id_str in cells.categories: 
                f.write(f"{cell_id_str}")
        print(f"Saved barcodes (cells) to: {barcodes_path}")
        print("Feature-cell matrix successfully generated.")

    except Exception as e:
        print(f"Error generating or saving feature-cell matrix: {e}")
        traceback.print_exc()

# Helper function to get image config (could be shared or adapted from visualize_gene_expression)
def _get_image_config(image_id, all_image_configs):
    """Fetches a specific image_configuration by its ID."""
    if image_id in all_image_configs:
        return all_image_configs[image_id]
    logger.error(f"Image ID '{image_id}' not found in provided image_configurations.")
    return None

def map_transcripts_for_task(task_config, all_image_configurations):
    """Maps transcripts to cells for a single task defined in the configuration."""
    task_id = task_config.get('task_id', 'Unknown Task')
    logger.info(f"Starting transcript mapping for task: {task_id}")
    logger.debug(f"Task configuration: {task_config}")

    # Get essential linking IDs
    source_image_id = task_config.get("source_image_id")
    source_param_set_id = task_config.get("source_param_set_id")

    if not source_image_id or not source_param_set_id:
        logger.error(f"Task '{task_id}': 'source_image_id' and 'source_param_set_id' are required. Skipping.")
        return False

    # Get the image configuration for the source image
    image_config = _get_image_config(source_image_id, all_image_configurations)
    if not image_config:
        logger.error(f"Task '{task_id}': Could not retrieve image configuration for image_id '{source_image_id}'. Skipping.")
        return False

    # --- Derive parameters --- 
    # 1. Derive scale_factor
    segmentation_opts = image_config.get("segmentation_options", {})
    rescaling_cfg = segmentation_opts.get("rescaling_config")
    derived_scale_factor = 1.0
    if rescaling_cfg and "scale_factor" in rescaling_cfg:
        sf_from_config = rescaling_cfg["scale_factor"]
        if isinstance(sf_from_config, (int, float)) and 0 < sf_from_config <= 1.0:
            derived_scale_factor = sf_from_config
    logger.info(f"  Task '{task_id}': Derived scale_factor: {derived_scale_factor}")

    # 2. Derive image_was_configured_for_tiling (effective is_tile status for source)
    tiling_params = segmentation_opts.get("tiling_parameters", {})
    image_was_configured_for_tiling = tiling_params.get("apply_tiling", False)
    logger.info(f"  Task '{task_id}': Source image configured for tiling: {image_was_configured_for_tiling}")

    # 3. Derive source_processing_unit_name if not provided
    derived_processing_unit_name = task_config.get("source_processing_unit_display_name")
    if not derived_processing_unit_name:
        if image_was_configured_for_tiling:
            logger.error(f"Task '{task_id}': 'source_processing_unit_display_name' is mandatory when the linked image_configuration (id: {source_image_id}) has apply_tiling=true, but was not provided. Skipping.")
            return False
        else:
            original_fname = image_config.get("original_image_filename")
            if not original_fname:
                logger.error(f"Task '{task_id}': Cannot derive source_processing_unit_name because 'original_image_filename' is missing in image_config '{source_image_id}'. Skipping.")
                return False
            if derived_scale_factor != 1.0:
                base, ext = os.path.splitext(os.path.basename(original_fname))
                scale_factor_str_file = str(derived_scale_factor).replace('.', '_')
                derived_processing_unit_name = f"{base}_scaled_{scale_factor_str_file}{ext}"
            else:
                derived_processing_unit_name = os.path.basename(original_fname)
            logger.info(f"  Task '{task_id}': Derived source_processing_unit_name: {derived_processing_unit_name}")
    
    # 4. Derive effective MPP values for the mask
    original_mpp_x = image_config.get("mpp_x")
    original_mpp_y = image_config.get("mpp_y")
    if original_mpp_x is None or original_mpp_y is None:
        logger.error(f"Task '{task_id}': Original mpp_x or mpp_y not found for image_id '{source_image_id}'. Skipping.")
        return False
    
    effective_mpp_x = original_mpp_x / derived_scale_factor if derived_scale_factor != 0 else original_mpp_x
    effective_mpp_y = original_mpp_y / derived_scale_factor if derived_scale_factor != 0 else original_mpp_y
    logger.info(f"  Task '{task_id}': Effective MPP of mask (x,y): ({effective_mpp_x:.4f}, {effective_mpp_y:.4f})")

    # --- Get remaining parameters from task_config ---
    transcripts_path_rel = task_config.get("input_transcripts_path")
    output_dir_rel = task_config.get("output_base_dir")
    output_prefix = task_config.get("output_prefix", "mapped_transcripts") # Default prefix
    mask_path_override = task_config.get("mask_path_override")

    if not transcripts_path_rel or not output_dir_rel:
        logger.error(f"Task '{task_id}': 'input_transcripts_path' and 'output_base_dir' are required. Skipping.")
        return False

    # Resolve paths
    transcripts_path_abs = transcripts_path_rel
    if not os.path.isabs(transcripts_path_abs):
        transcripts_path_abs = os.path.join(PROJECT_ROOT, transcripts_path_abs)
    
    output_dir_abs = output_dir_rel
    if not os.path.isabs(output_dir_abs):
        output_dir_abs = os.path.join(PROJECT_ROOT, output_dir_abs)
    os.makedirs(output_dir_abs, exist_ok=True)

    # Determine mask_path
    actual_mask_path = None
    if mask_path_override:
        actual_mask_path = mask_path_override
        if not os.path.isabs(actual_mask_path):
             actual_mask_path = os.path.join(PROJECT_ROOT, actual_mask_path)
        logger.info(f"  Task '{task_id}': Using overridden mask path: {actual_mask_path}")
    else:
        experiment_id_final = construct_full_experiment_id(
            image_id=source_image_id, 
            param_set_id=source_param_set_id, 
            scale_factor=derived_scale_factor,
            processing_unit_name_for_tile=derived_processing_unit_name if image_was_configured_for_tiling else None,
            is_tile=image_was_configured_for_tiling 
        )
        actual_mask_path = construct_mask_path(RESULTS_DIR_BASE, experiment_id_final, derived_processing_unit_name)
        logger.info(f"  Task '{task_id}': Derived mask path: {actual_mask_path}")

    if not actual_mask_path or not os.path.exists(actual_mask_path):
        logger.error(f"Mask file not found for task {task_id}. Path: {actual_mask_path}. Skipping.")
        return False

    logger.info(f"Using mask file: {actual_mask_path}")
    logger.info(f"Loading transcripts from: {transcripts_path_abs}")

    # Load data
    # Assuming load_transcripts and load_segmentation_mask are defined elsewhere in the file and use logger
    transcripts_df = load_transcripts(transcripts_path_abs, qv_threshold=task_config.get("qv_threshold", DEFAULT_QV_THRESHOLD))
    mask_image = load_segmentation_mask(actual_mask_path)

    if transcripts_df is None or mask_image is None:
        logger.error(f"Failed to load transcripts or mask for task {task_id}. Skipping.")
        return False

    # Perform mapping (assuming map_transcripts_to_cells is defined elsewhere and uses logger)
    # The map_transcripts_to_cells function now needs the effective MPP values.
    # It seems it might also need x/y_offset_microns if the transcript coordinates are global and mask is for a tile.
    # This needs careful review of how coordinates are handled for tiles vs whole images.
    # For now, assuming transcript x/y are relative to the specific mask origin, or global and offsets handled if needed.
    x_offset_microns = task_config.get("x_offset_microns_if_tile", 0) if image_was_configured_for_tiling else 0 # Example: tile specific offsets
    y_offset_microns = task_config.get("y_offset_microns_if_tile", 0) if image_was_configured_for_tiling else 0

    mapped_df = map_transcripts_to_cells(transcripts_df, mask_image, 
                                         microns_per_pixel_x=effective_mpp_x, 
                                         microns_per_pixel_y=effective_mpp_y,
                                         x_offset_microns=x_offset_microns,
                                         y_offset_microns=y_offset_microns)

    if mapped_df is None:
        logger.error(f"Transcript mapping failed for task {task_id}. Skipping further outputs for this task.")
        return False

    # Save mapped transcripts
    mapped_transcripts_output_path = os.path.join(output_dir_abs, f"{output_prefix}_mapped_transcripts.csv")
    try:
        mapped_df.to_csv(mapped_transcripts_output_path, index=False)
        logger.info(f"Saved mapped transcripts to: {mapped_transcripts_output_path}")
    except Exception as e:
        logger.error(f"Error saving mapped transcripts CSV for task {task_id}: {e}")
        # Continue to attempt feature-cell matrix generation if mapping was successful

    # Generate and save feature-cell matrix (assuming generate_feature_cell_matrix is defined elsewhere and uses logger)
    generate_feature_cell_matrix(mapped_df, output_dir_abs, filename_prefix=output_prefix)
    
    logger.info(f"Successfully completed transcript mapping for task: {task_id}")
    return True

def main():
    setup_logging() # Make sure logging is configured
    parser = argparse.ArgumentParser(description="Map transcripts to segmented cells based on parameter_sets.json.")
    parser.add_argument("--config", default="parameter_sets.json", help="Path to the JSON config file.")
    parser.add_argument("--task_id", default=None, help="Specific mapping task ID to run. Runs all active if not set.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        logger.critical(f"Configuration file not found: {config_path}"); return

    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
    except Exception as e:
        logger.critical(f"Error reading or parsing config file {config_path}: {e}"); return

    mapping_tasks = full_config.get("mapping_tasks", [])
    all_image_configs = {img_cfg["image_id"]: img_cfg for img_cfg in full_config.get("image_configurations", [])}

    if not mapping_tasks:
        logger.info("No mapping tasks found in configuration."); return
    if not all_image_configs:
        logger.error("No image_configurations found. Cannot process mapping tasks that need to derive parameters."); return

    tasks_to_process = []
    if args.task_id:
        task = next((t for t in mapping_tasks if t.get("task_id") == args.task_id), None)
        if task: tasks_to_process.append(task)
        else: logger.error(f"Mapping task ID '{args.task_id}' not found."); return
    else:
        tasks_to_process = [t for t in mapping_tasks if t.get("is_active", True)]

    if not tasks_to_process:
        logger.info("No active mapping tasks to run."); return

    success_count = 0
    for task_config in tasks_to_process:
        if map_transcripts_for_task(task_config, all_image_configs):
            success_count += 1
    
    logger.info(f"Finished processing all mapping tasks. Successful: {success_count}/{len(tasks_to_process)}.")

if __name__ == "__main__":
    # Example of how to call from command line (adjust paths and parameters in JSON):
    # python -m src.map_transcripts_to_cells --config parameter_sets.json --task_id map_experiment1_job1
    # python -m src.map_transcripts_to_cells --config parameter_sets.json 
    main()

