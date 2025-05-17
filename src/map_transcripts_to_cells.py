import os
import argparse
import pandas as pd
import numpy as np
import tifffile # Or from cellpose import io
from scipy.sparse import coo_matrix # For MTX, we need to write 3 files
import scipy.io # For saving MTX files
import traceback

# --- Default Values ---
DEFAULT_QV_THRESHOLD = 20.0

def load_transcripts(transcript_file, qv_threshold):
    """Loads transcripts, likely from a Parquet file, and filters by QV."""
    print(f"Loading transcripts from: {transcript_file} ...")
    try:
        if transcript_file.endswith(".parquet"):
            df = pd.read_parquet(transcript_file)
        elif transcript_file.endswith(".csv"):
            df = pd.read_csv(transcript_file)
        else:
            raise ValueError("Unsupported transcript file format. Please use .parquet or .csv.")
        
        print(f"Loaded {len(df)} total transcripts.")
        
        if 'qv' in df.columns:
            df_filtered = df[df['qv'] >= qv_threshold].copy()
            print(f"Filtered {len(df_filtered)} transcripts with QV >= {qv_threshold}.")
        else:
            print("Warning: 'qv' column not found in transcript data. No QV filtering applied.")
            df_filtered = df.copy()
            
        required_cols = ['x_location', 'y_location', 'feature_name'] 
        for col in required_cols:
            if col not in df_filtered.columns:
                raise ValueError(f"Missing required column '{col}' in transcript data.")
        if 'transcript_id' not in df_filtered.columns:
             print("Warning: 'transcript_id' not found. Using index as transcript_id.")
             df_filtered['transcript_id'] = df_filtered.index

        return df_filtered
    except Exception as e:
        print(f"Error loading or filtering transcripts: {e}")
        traceback.print_exc()
        return None

def load_segmentation_mask(mask_file_path):
    """Loads the integer-labeled segmentation mask."""
    print(f"Loading segmentation mask from: {mask_file_path} ...")
    try:
        from cellpose import io as cellpose_io 
        mask = cellpose_io.imread(mask_file_path)
        print(f"Loaded mask with shape: {mask.shape}, dtype: {mask.dtype}, max_id: {np.max(mask)}")
        return mask
    except Exception as e:
        print(f"Error loading segmentation mask: {e}")
        traceback.print_exc()
        return None

def map_transcripts_to_cells(transcripts_df, mask_image, microns_per_pixel_x, microns_per_pixel_y, 
                               x_offset_microns=0, y_offset_microns=0):
    """
    Assigns transcripts to cells based on their location within the segmentation mask.
    """
    print("Mapping transcripts to cells...")
    if transcripts_df is None or mask_image is None:
        return None

    transcripts_df['x_pixel'] = ((transcripts_df['x_location'] - x_offset_microns) / microns_per_pixel_x).astype(int)
    transcripts_df['y_pixel'] = ((transcripts_df['y_location'] - y_offset_microns) / microns_per_pixel_y).astype(int)
    
    mask_height, mask_width = mask_image.shape
    assigned_cell_ids = np.full(len(transcripts_df), -1, dtype=np.int32)

    valid_indices = (
        (transcripts_df['x_pixel'] >= 0) & (transcripts_df['x_pixel'] < mask_width) &
        (transcripts_df['y_pixel'] >= 0) & (transcripts_df['y_pixel'] < mask_height)
    )
    
    valid_transcripts_df = transcripts_df[valid_indices]
    print(f"Found {len(valid_transcripts_df)} transcripts within mask bounds out of {len(transcripts_df)}.")

    if not valid_transcripts_df.empty:
        assigned_ids_for_valid = mask_image[valid_transcripts_df['y_pixel'].values, 
                                            valid_transcripts_df['x_pixel'].values]
        assigned_cell_ids[valid_indices] = np.where(assigned_ids_for_valid > 0, assigned_ids_for_valid, -1)

    transcripts_df['assigned_cell_id'] = assigned_cell_ids
    
    num_assigned = np.sum(transcripts_df['assigned_cell_id'] > 0)
    print(f"Assigned {num_assigned} transcripts to cells based on segmentation mask.")
    return transcripts_df

def generate_feature_cell_matrix(mapped_transcripts_df, output_dir, filename_prefix="feature_cell_matrix"):
    """
    Generates a feature-cell matrix in MTX format.
    """
    print("Generating feature-cell matrix...")
    assigned_df = mapped_transcripts_df[mapped_transcripts_df['assigned_cell_id'] > 0].copy()

    if assigned_df.empty:
        print("No transcripts were assigned to cells. Cannot generate feature-cell matrix.")
        return

    assigned_df['assigned_cell_id'] = assigned_df['assigned_cell_id'].astype(str) 
    
    genes = pd.Categorical(assigned_df['feature_name'])
    cells = pd.Categorical(assigned_df['assigned_cell_id'])

    counts = coo_matrix((np.ones(len(assigned_df), dtype=np.int32), 
                         (genes.codes, cells.codes)), 
                        shape=(len(genes.categories), len(cells.categories)))

    matrix_path = os.path.join(output_dir, f"{filename_prefix}_matrix.mtx")
    genes_path = os.path.join(output_dir, f"{filename_prefix}_features.tsv")
    barcodes_path = os.path.join(output_dir, f"{filename_prefix}_barcodes.tsv")

    try:
        scipy.io.mmwrite(matrix_path, counts)
        print(f"Saved MTX matrix to: {matrix_path}")

        with open(genes_path, 'w') as f:
            for gene_name in genes.categories:
                f.write(f"{gene_name}	{gene_name}	Gene Expression")
        print(f"Saved features (genes) to: {genes_path}")

        with open(barcodes_path, 'w') as f:
            for cell_id in cells.categories:
                f.write(f"{cell_id}")
        print(f"Saved barcodes (cells) to: {barcodes_path}")
        print("Feature-cell matrix successfully generated.")

    except Exception as e:
        print(f"Error generating or saving feature-cell matrix: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Xenium transcripts to Cellpose segmented cells.")
    parser.add_argument("transcript_file", help="Path to the transcript data file (e.g., transcripts.parquet).")
    parser.add_argument("mask_file", help="Path to the Cellpose segmentation mask TIFF file (_mask.tif).")
    parser.add_argument("output_dir", help="Directory to save the mapped transcripts and feature-cell matrix.")
    
    parser.add_argument("--mpp_x", type=float, required=True, help="Microns per pixel in the X dimension of the mask image.")
    parser.add_argument("--mpp_y", type=float, required=True, help="Microns per pixel in the Y dimension of the mask image.")
    
    parser.add_argument("--x_offset", type=float, default=0.0, help="Offset in microns for X coordinates (transcript_x = mask_pixel_x * mpp_x + x_offset). Default: 0.")
    parser.add_argument("--y_offset", type=float, default=0.0, help="Offset in microns for Y coordinates (transcript_y = mask_pixel_y * mpp_y + y_offset). Default: 0.")

    parser.add_argument("--qv_threshold", type=float, default=DEFAULT_QV_THRESHOLD, 
                        help=f"Minimum Phred-scaled quality score (qv) for transcripts (default: {DEFAULT_QV_THRESHOLD}).")
    
    parser.add_argument("--output_prefix", type=str, default="cellpose_mapped", 
                        help="Prefix for output files (e.g., mapped transcripts CSV, feature-cell matrix prefix).")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print("--- Starting Transcript Mapping ---")
    transcripts = load_transcripts(args.transcript_file, args.qv_threshold)
    
    if transcripts is not None and not transcripts.empty:
        mask = load_segmentation_mask(args.mask_file)
        if mask is not None:
            mapped_df = map_transcripts_to_cells(transcripts, mask, 
                                                 args.mpp_x, args.mpp_y,
                                                 args.x_offset, args.y_offset)
            
            if mapped_df is not None:
                mapped_csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_transcripts_with_cell_ids.csv")
                try:
                    mapped_df.to_csv(mapped_csv_path, index=False)
                    print(f"Saved mapped transcripts with cell IDs to: {mapped_csv_path}")
                except Exception as e:
                    print(f"Error saving mapped transcripts CSV: {e}")

                generate_feature_cell_matrix(mapped_df, args.output_dir, filename_prefix=args.output_prefix)
            else:
                print("Transcript mapping step failed.")
        else:
            print("Mask loading failed. Cannot proceed with mapping.")
    else:
        print("Transcript loading or filtering failed, or no transcripts passed QV filter.")
    
    print("--- Transcript Mapping Finished ---")

