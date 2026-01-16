import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
import glob

class FastNeuralynxLoader:
    """Ultra-fast Neuralynx .ncs file loader optimized for your data structure"""
    
    @staticmethod
    def read_ncs_file(filepath, target_fs=128, decimate_factor=None):
        """
        Read a single .ncs file as fast as possible
        
        Parameters:
        -----------
        filepath : str
            Path to .ncs file
        target_fs : int
            Target sampling rate (default 128 Hz)
        decimate_factor : int
            Decimation factor (auto-calculated if None)
        """
        with open(filepath, 'rb') as f:
            # Skip header (16KB)
            f.seek(16384)
            
            # Read all data at once
            raw_data = f.read()
        
        # Parse header quickly to get sampling rate
        with open(filepath, 'rb') as f:
            header = f.read(16384).decode('latin-1', errors='ignore')
            
        # Extract sampling rate from header
        original_fs = 4096  # Default for macroelectrodes
        for line in header.split('\r\n'):
            if '-SamplingFrequency' in line:
                try:
                    original_fs = int(line.split()[-1])
                except:
                    pass
                break
        
        # Calculate number of records
        n_records = len(raw_data) // 1044
        
        if n_records == 0:
            return np.array([]), original_fs, target_fs
        
        # Pre-allocate array
        n_samples = n_records * 512
        data = np.empty(n_samples, dtype=np.int16)
        
        # Fast extraction: skip record headers, extract only samples
        for i in range(n_records):
            offset = i * 1044 + 20  # Skip 20-byte record header
            # Extract 512 samples (1024 bytes)
            data[i*512:(i+1)*512] = np.frombuffer(
                raw_data[offset:offset+1024], 
                dtype=np.int16
            )
        
        # Convert to microvolts (typical Neuralynx ADC conversion)
        data = data.astype(np.float32) * 0.061035  # Typical ADBitVolts value
        
        # Downsample
        if decimate_factor is None:
            decimate_factor = int(original_fs / target_fs)
        
        if decimate_factor > 1:
            # Fast decimation using slicing (faster than scipy for visualization)
            data = data[::decimate_factor]
        
        return data, original_fs, target_fs
    
    @staticmethod
    def extract_channel_name(filepath):
        """
        Extract channel name from filepath
        Example: 03415_2024-04-24_14-24_AmT2_7.ncs -> AmT2_7
        """
        filename = Path(filepath).stem
        # Pattern: {patient_id}_{date}_{time}_{channel_name}
        # Split by underscore and take everything after the time (index 2)
        parts = filename.split('_')
        if len(parts) >= 4:
            # Channel name is everything after the first 3 parts (patient_date_time)
            channel_name = '_'.join(parts[3:])
            return channel_name
        return filename
    
    @staticmethod
    def is_valid_macroelectrode(filepath, contact_range=(1, 5)):
        """
        Check if channel is a valid macroelectrode contact
        
        Rules:
        1. Channel name must contain '_' (e.g., AmT2_7)
        2. Contact number must be between contact_range (default 1-5)
        3. Channel name must NOT start with 'm' (microelectrode indicator, e.g., mAmT2_1)
        
        Parameters:
        -----------
        filepath : str
            Path to .ncs file
        contact_range : tuple
            (min_contact, max_contact) - default (1, 5)
        """
        channel_name = FastNeuralynxLoader.extract_channel_name(filepath)
        
        # Must contain underscore
        if '_' not in channel_name:
            return False
        
        # Exclude microelectrodes: channel name starts with 'm' (e.g., mAmT2_1)
        if channel_name.startswith('m'):
            return False
        
        # Extract contact number (last part after underscore)
        parts = channel_name.rsplit('_', 1)
        if len(parts) != 2:
            return False
        
        electrode_name, contact_str = parts
        
        # Contact must be a number
        try:
            contact_num = int(contact_str)
        except ValueError:
            return False
        
        # Contact must be in range
        min_contact, max_contact = contact_range
        if contact_num < min_contact or contact_num > max_contact:
            return False
        
        return True

    def load_folder(self, folder_path, target_fs=128, max_workers=None, 
                   contact_range=(1, 5), parallel_method='thread'):
        """
        Load all .ncs files from a folder in parallel
        
        Parameters:
        -----------
        folder_path : str
            Path to folder containing .ncs files
        target_fs : int
            Target sampling rate for downsampling
        max_workers : int
            Number of parallel workers (None = auto)
        contact_range : tuple
            (min, max) contact numbers to load (default (1, 5) for contacts 1-5)
        parallel_method : str
            'thread' or 'process' for parallel loading
        """
        # Find all .ncs files
        all_ncs_files = sorted(glob.glob(os.path.join(folder_path, '*.ncs')))
        
        if not all_ncs_files:
            raise ValueError(f"No .ncs files found in {folder_path}")
        
        print(f"Found {len(all_ncs_files)} .ncs files")
        
        # Filter for valid macroelectrode contacts
        ncs_files = [f for f in all_ncs_files if self.is_valid_macroelectrode(f, contact_range)]
        
        excluded = len(all_ncs_files) - len(ncs_files)
        print(f"  - {len(ncs_files)} valid macroelectrode contacts (range {contact_range[0]}-{contact_range[1]})")
        print(f"  - {excluded} channels excluded (no underscore, microelectrodes, or outside contact range)")
        
        if not ncs_files:
            raise ValueError("No valid macroelectrode files to load!")
        
        # Choose executor
        ExecutorClass = ThreadPoolExecutor if parallel_method == 'thread' else ProcessPoolExecutor
        
        # Load in parallel
        channel_names = []
        signals = []
        
        print(f"\nLoading {len(ncs_files)} channels...")
        with ExecutorClass(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.read_ncs_file, f, target_fs): f 
                for f in ncs_files
            }
            
            for i, future in enumerate(futures):
                filepath = futures[future]
                try:
                    data, orig_fs, final_fs = future.result()
                    if len(data) > 0:
                        channel_names.append(self.extract_channel_name(filepath))
                        signals.append(data)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(ncs_files):
                        print(f"  Loaded {i + 1}/{len(ncs_files)} channels...")
                        
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        print(f"\nSuccessfully loaded {len(signals)} channels")
        
        if not signals:
            raise ValueError("No data was loaded successfully!")
        
        # Stack signals (pad if necessary)
        max_len = max(len(s) for s in signals)
        data_matrix = np.zeros((len(signals), max_len), dtype=np.float32)
        
        for i, sig in enumerate(signals):
            data_matrix[i, :len(sig)] = sig
        
        return data_matrix, channel_names, target_fs