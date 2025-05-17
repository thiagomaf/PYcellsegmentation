import os
import argparse
import tifffile
import numpy as np
import traceback

def get_plane_from_multidim_data(image_data, axes_order, target_channel=None, target_z=None, target_time=None):
    """
    Extracts a 2D plane (Y, X) from multi-dimensional image data based on axes order.
    """
    axes = axes_order.upper()
    shape = image_data.shape
    ndim = image_data.ndim

    print(f"Attempting to extract plane. Original data shape: {shape}, axes: '{axes}'")
    print(f"Targets: C={target_channel}, Z={target_z}, T={target_time}")

    if 'Y' not in axes or 'X' not in axes:
        print(f"Error: Axes 'Y' and 'X' must be present in axes_order: '{axes_order}'")
        return None
    
    if target_channel is not None and 'C' not in axes:
        print(f"Warning: Target channel {target_channel} specified, but 'C' not in axes '{axes}'. Ignoring target_channel.")
        target_channel = None
    if target_z is not None and 'Z' not in axes:
        print(f"Warning: Target Z-plane {target_z} specified, but 'Z' not in axes '{axes}'. Ignoring target_z.")
        target_z = None
    if target_time is not None and 'T' not in axes:
        print(f"Warning: Target time point {target_time} specified, but 'T' not in axes '{axes}'. Ignoring target_time.")
        target_time = None

    slicing_tuple = [slice(None)] * ndim # CORRECTED VARIABLE NAME

    idx_y = -1
    idx_x = -1
    try:
        idx_y = axes.index('Y')
        idx_x = axes.index('X')
    except ValueError:
        print(f"Error: Could not find Y or X in axes string '{axes}'.")
        return None

    for i, axis_char in enumerate(axes):
        if axis_char == 'C':
            slicing_tuple[i] = target_channel if target_channel is not None else 0 
        elif axis_char == 'Z':
            slicing_tuple[i] = target_z if target_z is not None else 0 
        elif axis_char == 'T':
            slicing_tuple[i] = target_time if target_time is not None else 0 
        elif axis_char in ('Y', 'X'):
            slicing_tuple[i] = slice(None) 
        else: 
            print(f"Warning: Unknown axis '{axis_char}' in '{axes}'. Taking first slice for this dimension.")
            slicing_tuple[i] = 0
            
    try:
        selected_plane = image_data[tuple(slicing_tuple)] # CORRECTED VARIABLE NAME
    except IndexError as e:
        print(f"Error during slicing with {tuple(slicing_tuple)}: {e}") # CORRECTED VARIABLE NAME
        return None
    
    squeezed_plane = selected_plane.squeeze()
    if squeezed_plane.ndim == 2:
        print(f"Successfully extracted 2D plane. Shape: {squeezed_plane.shape}")
        return squeezed_plane
    else:
        print(f"Error: Extracted plane is not 2D after slicing and squeezing. Final shape: {squeezed_plane.shape}")
        print("  Original selected_plane shape before squeeze:", selected_plane.shape)
        print("  Slicing tuple used:", tuple(slicing_tuple)) # CORRECTED VARIABLE NAME
        return None


def process_ome_tiff(ome_tiff_path, output_dir, series_index=0, 
                     channel_index=None, z_index=None, time_index=None, 
                     output_prefix="processed"): 
    if not os.path.exists(ome_tiff_path):
        print(f"Error: Input OME-TIFF file not found: {ome_tiff_path}")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    try:
        with tifffile.TiffFile(ome_tiff_path) as tif:
            if not tif.series or series_index >= len(tif.series):
                print(f"Error: Series index {series_index} is out of bounds or no series found.")
                if tif.pages: 
                     print("Attempting to read as a simple TIFF (first page).")
                     if not tif.pages: print("No pages found in TIFF."); return
                     image_data = tif.pages[0].asarray()
                     axes_order = 'YX' 
                     if image_data.ndim > 2 and image_data.shape[-1] <=4 : 
                         axes_order = 'YXC' 
                     elif image_data.ndim ==3: 
                         axes_order = 'ZYX' 
                else:
                    print("No OME series and no pages found.")
                    return
            else:
                selected_series = tif.series[series_index]
                image_data = selected_series.asarray()
                axes_order = selected_series.axes
                print(f"Selected OME series {series_index}. Original data shape: {image_data.shape}, axes: '{axes_order}'")

            plane_2d = get_plane_from_multidim_data(image_data, axes_order, 
                                                    target_channel=channel_index, 
                                                    target_z=z_index, 
                                                    target_time=time_index)

            if plane_2d is not None:
                filename_parts = [output_prefix]
                if series_index is not None: filename_parts.append(f"s{series_index}")
                if channel_index is not None: filename_parts.append(f"c{channel_index}")
                if z_index is not None: filename_parts.append(f"z{z_index}")
                if time_index is not None: filename_parts.append(f"t{time_index}")
                
                output_filename = "_".join(filename_parts) + ".tif"
                output_filepath = os.path.join(output_dir, output_filename)

                tifffile.imwrite(output_filepath, plane_2d)
                print(f"Successfully saved processed 2D image to: {output_filepath}")
                print(f"  Dimensions: {plane_2d.shape}, Data type: {plane_2d.dtype}")
            else:
                print("Failed to extract a 2D plane from the OME-TIFF.")

    except Exception as e:
        print(f"An error occurred while processing {ome_tiff_path}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process OME-TIFF files for Cellpose pipeline.")
    parser.add_argument("input_ometiff", help="Path to the input OME-TIFF file.")
    parser.add_argument("output_dir", help="Directory to save the processed 2D TIFF files.")
    parser.add_argument("--series", type=int, default=0, help="Index of the series to process (default: 0).")
    parser.add_argument("--channel", type=int, default=None, help="Index of the channel to extract (e.g., for DAPI). Default: None (attempts to use first channel or assumes single channel if 'C' is not in axes).")
    parser.add_argument("--zplane", type=int, default=None, help="Index of the Z-plane to extract. Default: None (attempts to use first Z-plane or assumes 2D if 'Z' is not in axes).")
    parser.add_argument("--time", type=int, default=None, help="Index of the time point to extract. Default: None (attempts to use first time point or assumes no time dim if 'T' is not in axes).")
    parser.add_argument("--prefix", type=str, default="processed_plane", help="Prefix for the output TIFF filename (default: 'processed_plane').")

    args = parser.parse_args()

    print(f"Processing OME-TIFF: {args.input_ometiff}")
    print(f"Output directory: {args.output_dir}")
    
    process_ome_tiff(args.input_ometiff, args.output_dir, 
                     series_index=args.series, 
                     channel_index=args.channel, 
                     z_index=args.zplane, 
                     time_index=args.time,
                     output_prefix=args.prefix)
    print("Pre-processing finished.")

