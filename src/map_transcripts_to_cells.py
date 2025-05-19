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
from .pipeline_utils import setup_logging, construct_full_experiment_id, get_mask_path_from_experiment_id

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

def map_transcripts_for_task(task_config):
    """Maps transcripts to cells for a single task defined in the configuration."""
    logger.info(f"Starting transcript mapping for task: {task_config.get('task_id', 'Unknown Task')}")
    logger.debug(f"Task configuration: {task_config}")

    mpp_x = task_config.get("mpp_x_of_mask")
    mpp_y = task_config.get("mpp_y_of_mask")
    if mpp_x is None or mpp_y is None:
        logger.error(f"MPP values (mpp_x_of_mask, mpp_y_of_mask) are required for task {task_config.get('task_id')}. Skipping.")
        return False

    transcripts_path = task_config.get("input_transcripts_path")
    if not os.path.isabs(transcripts_path):
        transcripts_path = os.path.join(PROJECT_ROOT, transcripts_path)
    
    output_dir = task_config.get("output_base_dir")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_prefix = task_config.get("output_prefix", "mapped_transcripts")

    mask_path = task_config.get("mask_path_override")
    if not mask_path:
        # Derive mask path
        source_image_id = task_config.get("source_image_id")
        source_param_set_id = task_config.get("source_param_set_id")
        scale_factor = task_config.get("source_segmentation_scale_factor")
        # Use 'source_processing_unit_display_name' which should be the actual filename of the segmented unit (e.g. image_scaled_0.5.tif or tile_r0_c0.tif)
        processing_unit_display_name = task_config.get("source_processing_unit_display_name")
        is_tile = task_config.get("source_segmentation_is_tile", False) # New field needed to distinguish tile from full image name

        if not source_image_id or not source_param_set_id or not processing_unit_display_name:
            logger.error(f"Task {task_config.get('task_id')}: For derived mask path, 'source_image_id', 'source_param_set_id', and 'source_processing_unit_display_name' are required. Skipping.")
            return False
        
        experiment_id_final = construct_full_experiment_id(
            image_id=source_image_id, 
            param_set_id=source_param_set_id, 
            scale_factor=scale_factor,
            # If it's a tile, processing_unit_display_name is the tile name itself (e.g. tile_r0_c0.tif)
            # If not a tile, this argument to construct_full_experiment_id should be None.
            processing_unit_name_for_tile=processing_unit_display_name if is_tile else None,
            is_tile=is_tile 
        )
        
        # The `processing_unit_display_name` is the direct name for mask lookup (e.g., my_image_scaled_0_5.tif or tile_r0_c0.tif)
        mask_path = get_mask_path_from_experiment_id(RESULTS_DIR_BASE, experiment_id_final, processing_unit_display_name)

    if not mask_path or not os.path.exists(mask_path):
        logger.error(f"Mask file not found for task {task_config.get('task_id')}. Path: {mask_path}. Searched based on source parameters. Skipping.")
        return False

    logger.info(f"Using mask file: {mask_path}")
    logger.info(f"Loading transcripts from: {transcripts_path}")

    try:
        mask = tifffile.imread(mask_path)
        logger.info(f"Mask loaded successfully. Shape: {mask.shape}, Max ID: {mask.max()}")

        if transcripts_path.endswith('.parquet'):
            df = read_table(transcripts_path).to_pandas()
        elif transcripts_path.endswith('.csv'):
            df = pd.read_csv(transcripts_path)
        else:
            logger.error(f"Unsupported transcript file format: {transcripts_path}. Must be .parquet or .csv. Skipping task.")
            return False
        logger.info(f"Transcripts loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")

        required_cols = ['global_x', 'global_y', 'gene'] 
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Transcript data at {transcripts_path} is missing one or more required columns: {required_cols}. Found: {df.columns.tolist()}. Skipping task.")
            return False

        df.rename(columns={'global_x': 'pixel_x', 'global_y': 'pixel_y'}, inplace=True)
        logger.info("Assuming 'global_x' and 'global_y' in transcript file are pixel coordinates at the mask's resolution.")

        df_filtered = df[(df['pixel_x'] >= 0) & (df['pixel_x'] < mask.shape[1]) &
                         (df['pixel_y'] >= 0) & (df['pixel_y'] < mask.shape[0])].copy()
        
        num_filtered_out = len(df) - len(df_filtered)
        if num_filtered_out > 0:
            logger.info(f"Filtered out {num_filtered_out} transcripts that were outside the mask boundaries.")

        if df_filtered.empty:
            logger.warning("No transcripts remaining after filtering for mask boundaries. Proceeding to write empty outputs.")
            cell_ids_for_transcripts = []
        else:
            cell_ids_for_transcripts = mask[df_filtered['pixel_y'].astype(int).values, 
                                          df_filtered['pixel_x'].astype(int).values]
        
        df_filtered.loc[:, 'cell_id'] = cell_ids_for_transcripts
        logger.info(f"Assigned cell IDs to {len(df_filtered[df_filtered['cell_id'] > 0])} transcripts within cells.")

        mapped_transcripts_output_path = os.path.join(output_dir, f"{output_prefix}_transcripts_mapped.csv")
        df_filtered.to_csv(mapped_transcripts_output_path, index=False)
        logger.info(f"Mapped transcripts saved to: {mapped_transcripts_output_path}")

        df_in_cells = df_filtered[df_filtered['cell_id'] > 0]
        if not df_in_cells.empty:
            feature_cell_matrix = pd.crosstab(df_in_cells['gene'], df_in_cells['cell_id'])
            matrix_output_path = os.path.join(output_dir, f"{output_prefix}_feature_cell_matrix.csv")
            feature_cell_matrix.to_csv(matrix_output_path)
            logger.info(f"Feature-by-cell matrix saved to: {matrix_output_path}")
        else:
            logger.info("No transcripts were assigned to cells (cell_id > 0). Skipping feature-by-cell matrix generation.")
        
        logger.info(f"Transcript mapping for task {task_config.get('task_id', 'Unknown Task')} completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error during transcript mapping for task {task_config.get('task_id', 'Unknown Task')}: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Maps transcripts to segmented cells based on a mask.")
    parser.add_argument("--config", default="parameter_sets.json",
                        help="Path to the parameter sets JSON file (default: parameter_sets.json relative to project root).")
    parser.add_argument("--task_id", type=str, default=None,
                        help="Specific task_id from the mapping_tasks in the config file to run. If not provided, all active tasks will be run.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Logging level (default: INFO).")
    
    args = parser.parse_args()
    # Determine log level: CLI > JSON (if we add it to mapping_tasks or global) > default in setup_logging
    # For now, CLI or direct default in setup_logging if not passed.
    # We could enhance this to read from global_segmentation_settings.default_log_level if args.log_level is not set by CLI.
    effective_log_level = args.log_level.upper()
    # Potentially load from JSON config if args.log_level is None and a global default exists
    config_file_path_for_log = args.config
    if not os.path.isabs(config_file_path_for_log):
        config_file_path_for_log = os.path.join(PROJECT_ROOT, config_file_path_for_log)
    
    json_log_level = None
    if os.path.exists(config_file_path_for_log):
        try:
            with open(config_file_path_for_log, 'r') as f_log_check:
                log_check_config = json.load(f_log_check)
            global_settings_log = log_check_config.get("global_segmentation_settings", {})
            json_log_level = global_settings_log.get("default_log_level")
        except Exception:
            pass # Ignore if config can't be read for log level pre-setup

    if args.log_level:
         effective_log_level = args.log_level.upper()
    elif json_log_level and isinstance(json_log_level, str) and json_log_level.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        effective_log_level = json_log_level.upper()
    else:
        effective_log_level = "INFO" # Fallback default

    setup_logging(effective_log_level)
    logger = logging.getLogger(__name__) # Re-initialize logger with the correct module name after setup

    logger.info(f"Effective logging level set to: {effective_log_level}")

    config_file_path = args.config
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(PROJECT_ROOT, config_file_path)

    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found: {config_file_path}")
        return

    try:
        with open(config_file_path, 'r') as f:
            full_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading or parsing configuration file {config_file_path}: {e}")
        return

    mapping_tasks_config = full_config.get("mapping_tasks")
    if not mapping_tasks_config or not isinstance(mapping_tasks_config, list):
        logger.error("'mapping_tasks' section is missing or not a list in the configuration file.")
        return

    tasks_to_run = []
    if args.task_id:
        task_found = False
        for task_params in mapping_tasks_config:
            if task_params.get("task_id") == args.task_id:
                if task_params.get("is_active", False):
                    tasks_to_run.append(task_params)
                    task_found = True
                else:
                    logger.warning(f"Task {args.task_id} found but is not active. Skipping.")
                    return # Exit if specific task is inactive
                break
        if not task_found:
            logger.error(f"Specified task_id '{args.task_id}' not found in mapping_tasks.")
            return
    else:
        tasks_to_run = [task for task in mapping_tasks_config if task.get("is_active", False)]
        if not tasks_to_run:
            logger.info("No active mapping tasks found to run.")
            return

    logger.info(f"Found {len(tasks_to_run)} mapping task(s) to run.")
    successful_tasks = 0
    for i, task_params in enumerate(tasks_to_run):
        logger.info(f"--- Running mapping task {i+1}/{len(tasks_to_run)}: {task_params.get('task_id', 'Unnamed Task')} ---")
        if map_transcripts_for_task(task_params):
            successful_tasks += 1
    
    logger.info(f"--- Mapping Summary ---")
    logger.info(f"Total tasks processed: {len(tasks_to_run)}")
    logger.info(f"Successfully completed tasks: {successful_tasks}")
    logger.info(f"Failed tasks: {len(tasks_to_run) - successful_tasks}")

if __name__ == '__main__':
    # Example of how to call from command line (adjust paths and parameters in JSON):
    # python -m src.map_transcripts_to_cells --config parameter_sets.json --task_id map_experiment1_job1
    # python -m src.map_transcripts_to_cells --config parameter_sets.json 
    main()

