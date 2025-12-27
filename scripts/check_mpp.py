import json
import os
import tifffile
import xml.etree.ElementTree as ET

def check_mpp():
    config_path = "config/xenium_HE/HE/processing_config_HE_all.json"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    image_configs = config.get("image_configurations", [])
    print(f"Checking {len(image_configs)} images...")

    for img_conf in image_configs:
        image_id = img_conf.get("image_id")
        filepath = img_conf.get("original_image_filename")
        config_mpp_x = img_conf.get("mpp_x")
        config_mpp_y = img_conf.get("mpp_y")

        # Handle path separators
        filepath = filepath.replace("\\", "/")
        if not os.path.exists(filepath):
            print(f"[{image_id}] File not found: {filepath}")
            continue

        try:
            with tifffile.TiffFile(filepath) as tif:
                calculated_mpp_x = None
                calculated_mpp_y = None
                
                # Method 1: OME-XML
                ome_metadata = tif.ome_metadata
                if ome_metadata:
                    # Simple parsing of OME-XML to find PhysicalSizeX
                    # Note: OME-XML can be complex, this is a basic attempt
                    try:
                        root = ET.fromstring(ome_metadata)
                        # Namespace handling usually required, but try ignoring or wildcards
                        # Use a namespace-agnostic search if possible or just handle the common one
                        namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                        # Iterate all elements to find Pixels
                        for elem in root.iter():
                            if elem.tag.endswith('Pixels'):
                                phys_x = elem.get('PhysicalSizeX')
                                phys_y = elem.get('PhysicalSizeY')
                                unit_x = elem.get('PhysicalSizeXUnit', 'µm') # Default to microns if not specified
                                unit_y = elem.get('PhysicalSizeYUnit', 'µm')

                                if phys_x:
                                    # Convert to microns if needed (basic support)
                                    if unit_x == 'µm': calculated_mpp_x = float(phys_x)
                                if phys_y:
                                    if unit_y == 'µm': calculated_mpp_y = float(phys_y)
                                break
                    except Exception as e:
                        print(f"[{image_id}] OME-XML parsing error: {e}")

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
                            if unit_val == 2: # Inch
                                calculated_mpp_x = 25400 / x_res_float
                                calculated_mpp_y = 25400 / y_res_float
                            elif unit_val == 3: # Centimeter
                                calculated_mpp_x = 10000 / x_res_float
                                calculated_mpp_y = 10000 / y_res_float

                # Report
                if calculated_mpp_x is not None:
                    # Allow small floating point difference
                    diff_x = abs(calculated_mpp_x - config_mpp_x)
                    diff_y = abs(calculated_mpp_y - config_mpp_y)
                    
                    if diff_x > 0.001 or diff_y > 0.001:
                        print(f"[{image_id}] MISMATCH:")
                        print(f"  Config: x={config_mpp_x}, y={config_mpp_y}")
                        print(f"  Calculated: x={calculated_mpp_x:.4f}, y={calculated_mpp_y:.4f}")
                        print(f"  Difference: x={diff_x:.4f}, y={diff_y:.4f}")
                    else:
                        # pass # print(f"[{image_id}] Match ({calculated_mpp_x:.4f})")
                        print(f"[{image_id}] Match: Config={config_mpp_x}, Calculated={calculated_mpp_x:.4f}")
                else:
                    print(f"[{image_id}] Could not calculate MPP from metadata.")

        except Exception as e:
            print(f"[{image_id}] Error processing file: {e}")

if __name__ == "__main__":
    check_mpp()

