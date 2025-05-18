import os
import argparse
import json
import numpy as np
import tifffile 
from cellpose import io as cellpose_io 

RESULTS_DIR_BASE = "results"
RUN_LOG_FILE = os.path.join(RESULTS_DIR_BASE, "run_log.json")

def get_original_image_dims_from_log(run_log_data, original_image_id_to_stitch, param_set_id_to_stitch):
    """
    Attempts to infer original image dimensions from the first relevant tile's manifest info stored in the run log.
    This assumes the tiling manifest (and thus original dimensions) are consistent for all tiles of an original image.
    """
    for job in run_log_data:
        if (job.get("original_image_id_for_log") == original_image_id_to_stitch and 
            job.get("param_set_id_for_log") == param_set_id_to_stitch and 
            job.get("is_tile_for_log") and 
            job.get("tile_details_for_log") and 
            job["tile_details_for_log"].get("path")):
            
            # Attempt to find the manifest associated with these tiles
            # The manifest path might be inferred if tiles are stored in a structured way
            # e.g. images/tiled_outputs/<original_image_id>_tiles/<tile_prefix>_manifest.json
            # This part is tricky as the exact manifest path isn't directly in each tile's job log.
            # For now, we assume the first tile's manifest (if we could find it) would have the info.
            # A more robust way: the tiling script should save its manifest to a predictable location, 
            # or the segmentation pipeline could log the manifest path for each tiled run.
            
            # Fallback: Reconstruct from max extents of tiles if manifest not easily found
            tile_manifest_path_guess = None
            if job.get("tile_details_for_log") and job["tile_details_for_log"].get("path"):
                tile_dir = os.path.dirname(job["tile_details_for_log"]["path"])
                # Try to find a manifest in that directory based on a common prefix pattern
                # This is a guess, depends on how tile_large_image names manifests
                # E.g. if prefix was 'tile', manifest is 'tile_manifest.json'
                # We need the prefix used during tiling. This info isn't in the run_log per job easily.

                # Let's assume for now we can get it by iterating all tiles as before
                max_x_end = 0
                max_y_end = 0
                found_any_tile = False
                for check_job in run_log_data:
                     if (check_job.get("original_image_id_for_log") == original_image_id_to_stitch and 
                         check_job.get("param_set_id_for_log") == param_set_id_to_stitch and 
                         check_job.get("is_tile_for_log")):
                        found_any_tile = True
                        tile_info = check_job.get("tile_details_for_log", {})
                        x_start = tile_info.get("x_start_in_original", 0)
                        y_start = tile_info.get("y_start_in_original", 0)
                        width = tile_info.get("width_tile", 0)
                        height = tile_info.get("height_tile", 0)
                        max_x_end = max(max_x_end, x_start + width)
                        max_y_end = max(max_y_end, y_start + height)
                if found_any_tile:
                    return int(max_y_end), int(max_x_end)

    print(f"Warning: Could not determine original image dimensions for '{original_image_id_to_stitch}' from run log.")
    return None, None

def stitch_tile_masks(run_log_path, original_image_id_to_stitch, param_set_id_to_stitch, output_dir, output_filename):
    if not os.path.exists(run_log_path):
        print(f"Error: Run log file not found: {run_log_path}"); return

    try:
        with open(run_log_path, 'r') as f: run_log_data = json.load(f)
    except Exception as e: print(f"Error reading run log {run_log_path}: {e}"); return

    relevant_jobs = []
    for job in run_log_data:
        if (job.get("status") == "succeeded" and 
            job.get("is_tile_for_log") is True and 
            job.get("original_image_id_for_log") == original_image_id_to_stitch and 
            job.get("param_set_id_for_log") == param_set_id_to_stitch):
            relevant_jobs.append(job)

    if not relevant_jobs:
        print(f"No successful tile jobs found for OriginalImageID='{original_image_id_to_stitch}', ParamSetID='{param_set_id_to_stitch}'."); return

    print(f"Found {len(relevant_jobs)} tile masks to stitch for {original_image_id_to_stitch} with params {param_set_id_to_stitch}.")

    orig_height, orig_width = get_original_image_dims_from_log(run_log_data, original_image_id_to_stitch, param_set_id_to_stitch)

    if orig_height is None or orig_width is None or orig_height == 0 or orig_width == 0:
        print("Error: Could not determine original image dimensions for stitching. Cannot proceed."); return

    print(f"Canvas for stitched mask: H={orig_height}, W={orig_width}")
    stitched_mask = np.zeros((orig_height, orig_width), dtype=np.uint32)
    current_max_cell_id = 0

    for job_info in relevant_jobs:
        mask_path = job_info.get("output_mask_path")
        tile_details = job_info.get("tile_details_for_log")

        if not mask_path or not tile_details or not os.path.exists(mask_path):
            print(f"Warning: Skipping job {job_info.get('experiment_id_final')} due to missing mask or tile details."); continue
            
        print(f"  Processing tile: {tile_details['filename']} from job {job_info.get('experiment_id_final')}")
        tile_mask = cellpose_io.imread(mask_path)
        
        x_orig = tile_details["x_start_in_original"]
        y_orig = tile_details["y_start_in_original"]
        h_tile, w_tile = tile_mask.shape

        unique_ids_in_tile = np.unique(tile_mask)
        unique_ids_in_tile = unique_ids_in_tile[unique_ids_in_tile > 0]
        if len(unique_ids_in_tile) == 0: continue

        re_labeled_tile_mask = np.zeros_like(tile_mask, dtype=np.uint32)
        max_id_in_this_tile_after_relabel = 0
        for old_id in unique_ids_in_tile:
            new_id = current_max_cell_id + old_id
            re_labeled_tile_mask[tile_mask == old_id] = new_id
            if new_id > max_id_in_this_tile_after_relabel : max_id_in_this_tile_after_relabel = new_id
        
        target_y_slice = slice(y_orig, min(y_orig + h_tile, orig_height))
        target_x_slice = slice(x_orig, min(x_orig + w_tile, orig_width))
        
        # Ensure slices match the shape of re_labeled_tile_mask that we copy
        # This means re_labeled_tile_mask might need to be cropped if it extends beyond canvas
        # Or, more simply, ensure the part of re_labeled_tile_mask we copy fits into the slice on stitched_mask
        h_to_copy = target_y_slice.stop - target_y_slice.start
        w_to_copy = target_x_slice.stop - target_x_slice.start
        
        region_to_update = stitched_mask[target_y_slice, target_x_slice]
        tile_part_to_copy = re_labeled_tile_mask[:h_to_copy, :w_to_copy]
        
        new_cell_pixels_in_tile = tile_part_to_copy > 0
        region_to_update[new_cell_pixels_in_tile] = tile_part_to_copy[new_cell_pixels_in_tile]
        stitched_mask[target_y_slice, target_x_slice] = region_to_update
        
        current_max_cell_id = max(current_max_cell_id, max_id_in_this_tile_after_relabel)

    if not os.path.exists(output_dir):
        try: os.makedirs(output_dir)
        except OSError as e: print(f"Error creating output directory {output_dir}: {e}"); return

    output_filepath = os.path.join(output_dir, output_filename)
    try:
        save_dtype = np.uint16 if current_max_cell_id < 65535 else np.uint32
        tifffile.imwrite(output_filepath, stitched_mask.astype(save_dtype))
        print(f"Stitched mask saved to: {output_filepath}")
        print(f"  Total unique cell IDs in stitched mask: {current_max_cell_id}")
    except Exception as e:
        print(f"Error saving stitched mask: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch tiled Cellpose segmentation masks.")
    parser.add_argument("original_image_id", help="The 'image_id' from image_configurations in parameter_sets.json that was tiled.")
    parser.add_argument("param_set_id", help="The 'param_set_id' from cellpose_parameter_configurations used for segmentation.")
    parser.add_argument("output_dir", help="Directory to save the stitched mask.")
    parser.add_argument("--output_filename", default="stitched_mask.tif", help="Filename for the output stitched mask (default: stitched_mask.tif).")
    parser.add_argument("--run_log", default=RUN_LOG_FILE, help=f"Path to the run_log.json file (default: {RUN_LOG_FILE}).")
    
    args = parser.parse_args()

    print(f"Attempting to stitch masks for original image ID: '{args.original_image_id}' with param set ID: '{args.param_set_id}'")
    
    stitch_tile_masks(args.run_log, args.original_image_id, args.param_set_id, args.output_dir, args.output_filename)
    
    print("Stitching process finished.")

