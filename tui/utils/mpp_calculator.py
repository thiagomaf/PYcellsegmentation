"""MPP (microns per pixel) calculation utility from image metadata."""
import os
import tifffile
import xml.etree.ElementTree as ET
from typing import Tuple, Optional


def calculate_mpp_from_image(filepath: str) -> Tuple[float | None, float | None]:
    """
    Calculate MPP (microns per pixel) from image metadata.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Tuple of (mpp_x, mpp_y) or (None, None) if calculation fails
    """
    # Normalize path separators
    filepath = filepath.replace("\\", "/")
    
    if not os.path.exists(filepath):
        return (None, None)
    
    try:
        with tifffile.TiffFile(filepath) as tif:
            calculated_mpp_x = None
            calculated_mpp_y = None
            
            # Method 1: OME-XML
            ome_metadata = tif.ome_metadata
            if ome_metadata:
                try:
                    root = ET.fromstring(ome_metadata)
                    # Iterate all elements to find Pixels
                    for elem in root.iter():
                        if elem.tag.endswith('Pixels'):
                            phys_x = elem.get('PhysicalSizeX')
                            phys_y = elem.get('PhysicalSizeY')
                            unit_x = elem.get('PhysicalSizeXUnit', 'µm')  # Default to microns
                            unit_y = elem.get('PhysicalSizeYUnit', 'µm')
                            
                            if phys_x:
                                # Convert to microns if needed (basic support)
                                if unit_x == 'µm':
                                    calculated_mpp_x = float(phys_x)
                            if phys_y:
                                if unit_y == 'µm':
                                    calculated_mpp_y = float(phys_y)
                            break
                except Exception:
                    # OME-XML parsing failed, try TIFF tags
                    pass
            
            # Method 2: TIFF Tags (if OME failed or yielded nothing)
            if calculated_mpp_x is None:
                page = tif.pages[0]
                tags = page.tags
                
                x_res = tags.get('XResolution')
                y_res = tags.get('YResolution')
                res_unit = tags.get('ResolutionUnit')
                
                if x_res and y_res and res_unit:
                    x_val = x_res.value
                    y_val = y_res.value
                    unit_val = res_unit.value
                    
                    # x_val is (numerator, denominator)
                    if isinstance(x_val, tuple):
                        x_res_float = x_val[0] / x_val[1] if x_val[1] != 0 else 0
                    else:
                        x_res_float = x_val
                        
                    if isinstance(y_val, tuple):
                        y_res_float = y_val[0] / y_val[1] if y_val[1] != 0 else 0
                    else:
                        y_res_float = y_val
                    
                    if x_res_float > 0 and y_res_float > 0:
                        # Convert resolution to DPI first, then calculate MPP
                        # MPP = 25400 / DPI (where 25400 microns = 1 inch)
                        if unit_val == 2:  # Inches - resolution is already DPI
                            dpi_x = x_res_float
                            dpi_y = y_res_float
                        elif unit_val == 3:  # Centimeters - convert to DPI
                            dpi_x = x_res_float * 2.54
                            dpi_y = y_res_float * 2.54
                        else:
                            # Unknown unit, skip
                            dpi_x = None
                            dpi_y = None
                        
                        if dpi_x and dpi_y:
                            calculated_mpp_x = 25400.0 / dpi_x
                            calculated_mpp_y = 25400.0 / dpi_y
            
            return (calculated_mpp_x, calculated_mpp_y)
    
    except Exception:
        return (None, None)


def get_image_dimensions(filepath: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Get image dimensions (width, height) in pixels.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Tuple of (width, height) or (None, None) if reading fails
    """
    # Normalize path separators
    filepath = filepath.replace("\\", "/")
    
    if not os.path.exists(filepath):
        return (None, None)
    
    try:
        with tifffile.TiffFile(filepath) as tif:
            # Get first page dimensions
            page = tif.pages[0]
            # Shape is typically (height, width) or (depth, height, width) for multi-channel
            shape = page.shape
            
            if len(shape) >= 2:
                # For 2D: shape is (height, width)
                # For 3D+: shape is (depth, height, width) or (height, width, channels)
                # Take the last two dimensions as height and width
                height, width = shape[-2], shape[-1]
                return (width, height)
            else:
                return (None, None)
    
    except Exception:
        return (None, None)

