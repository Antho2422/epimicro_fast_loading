from core import FastNeuralynxLoader
import os
import glob

def print_channel_info(folder_path, contact_range=(1, 5)):
    """Print information about channels in the folder"""
    loader = FastNeuralynxLoader()
    all_ncs_files = sorted(glob.glob(os.path.join(folder_path, '*.ncs')))
    
    print(f"\nAnalyzing folder: {folder_path}")
    print(f"Total .ncs files: {len(all_ncs_files)}\n")
    
    valid_channels = []
    excluded_no_underscore = []
    excluded_out_of_range = []
    
    for f in all_ncs_files:
        channel = loader.extract_channel_name(f)
        
        # Check if has underscore
        if '_' not in channel:
            excluded_no_underscore.append(channel)
            continue
        
        # Check if in contact range
        if not loader.is_valid_macroelectrode(f, contact_range):
            excluded_out_of_range.append(channel)
            continue
        
        valid_channels.append(channel)
    
    print(f"Valid macroelectrode contacts (range {contact_range[0]}-{contact_range[1]}): {len(valid_channels)}")
    for ch in sorted(valid_channels):
        print(f"  ✓ {ch}")
    
    print(f"\nExcluded - No underscore ({len(excluded_no_underscore)}):")
    for ch in sorted(excluded_no_underscore)[:20]:  # Show first 20
        print(f"  ✗ {ch}")
    if len(excluded_no_underscore) > 20:
        print(f"  ... and {len(excluded_no_underscore) - 20} more")
    
    print(f"\nExcluded - Outside contact range ({len(excluded_out_of_range)}):")
    for ch in sorted(excluded_out_of_range)[:20]:  # Show first 20
        print(f"  ✗ {ch}")
    if len(excluded_out_of_range) > 20:
        print(f"  ... and {len(excluded_out_of_range) - 20} more")