import os
import argparse
import pandas as pd
import numpy as np
import tifffile # Or from cellpose import io
from scipy.sparse import coo_matrix
import scipy.io 
import traceback
from cellpose import io as cellpose_io # Using cellpose.io for consistency

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Xenium transcripts to Cellpose segmented cells.")
    parser.add_argument("transcript_file", help="Path to the transcript data file (e.g., transcripts.parquet or .csv).")
    parser.add_argument("mask_file", help="Path to the Cellpose segmentation mask TIFF file (_mask.tif).")
    parser.add_argument("output_dir", help="Directory to save the mapped transcripts and feature-cell matrix.")
    
    parser.add_argument("--mpp_x", type=float, required=True, help="Microns per pixel in X (mask image).")
    parser.add_argument("--mpp_y", type=float, required=True, help="Microns per pixel in Y (mask image).")
    
    parser.add_argument("--x_offset_microns", type=float, default=0.0, help="Offset in microns for X coordinates. Default: 0.")
    parser.add_argument("--y_offset_microns", type=float, default=0.0, help="Offset in microns for Y coordinates. Default: 0.")

    parser.add_argument("--qv_threshold", type=float, default=DEFAULT_QV_THRESHOLD, 
                        help=f"Min QV score for transcripts (default: {DEFAULT_QV_THRESHOLD}).")
    
    parser.add_argument("--output_prefix", type=str, default="mapped_transcripts", 
                        help="Prefix for output files.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"Error creating output directory {args.output_dir}: {e}")
            exit(1)


    print("--- Starting Transcript Mapping ---")
    transcripts_loaded = load_transcripts(args.transcript_file, args.qv_threshold)
    
    if transcripts_loaded is not None and not transcripts_loaded.empty:
        segmentation_mask = load_segmentation_mask(args.mask_file)
        if segmentation_mask is not None:
            mapped_transcripts_df = map_transcripts_to_cells(
                transcripts_loaded, segmentation_mask, 
                args.mpp_x, args.mpp_y,
                args.x_offset_microns, args.y_offset_microns
            )
            
            if mapped_transcripts_df is not None:
                mapped_csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_with_cell_ids.csv")
                try:
                    mapped_transcripts_df.to_csv(mapped_csv_path, index=False)
                    print(f"Saved mapped transcripts with cell IDs to: {mapped_csv_path}")
                except Exception as e:
                    print(f"Error saving mapped transcripts CSV: {e}")

                generate_feature_cell_matrix(mapped_transcripts_df, args.output_dir, 
                                             filename_prefix=args.output_prefix)
            else:
                print("Transcript mapping step returned None or empty DataFrame.")
        else:
            print("Mask loading failed. Cannot proceed with mapping.")
    else:
        print("Transcript loading or filtering failed, or no transcripts passed QV filter.")
    
    print("--- Transcript Mapping Finished ---")

