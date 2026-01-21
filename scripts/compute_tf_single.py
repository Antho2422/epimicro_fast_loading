#!/usr/bin/env python3
"""
Compute time-frequency map for a single EEG session.

This script is designed to be called from Slurm jobs. It processes
a single session and saves the TF map to disk as a .npz file.

Usage:
    python scripts/compute_tf_single.py --session-path /path/to/session --output-dir /path/to/output
    python scripts/compute_tf_single.py --index 42 --session-file sessions.txt --output-dir /path/to/output
"""

import argparse
import sys
import time
import os
import numpy as np
from pathlib import Path
from scipy.signal import spectrogram, decimate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import FastNeuralynxLoader


def compute_tf_map(data, fs, freq_range=(1, 100)):
    """
    Compute time-frequency map for a single session.
    
    Parameters
    ----------
    data : array (n_channels, n_samples)
        EEG data
    fs : float
        Sampling frequency
    freq_range : tuple
        (min_freq, max_freq) in Hz
    
    Returns
    -------
    freqs : array
        Frequency bins
    times : array
        Time bins
    Sxx : array
        Power spectral density (freq x time)
    duration : float
        Total duration in seconds
    """
    # Average all channels for overview
    overview_signal = np.mean(data, axis=0)
    
    # Downsample if fs > 512 Hz for faster computation
    effective_fs = fs
    if fs > 512:
        factor = int(fs // 256)
        overview_signal = decimate(overview_signal, factor)
        effective_fs = fs / factor
    
    # Compute spectrogram with coarse time resolution for overview
    nperseg = int(effective_fs * 2)
    noverlap = int(effective_fs * 1)
    
    f, t, Sxx = spectrogram(
        overview_signal,
        fs=effective_fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(256, nperseg),
    )
    
    # Keep only specified frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    freqs = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    
    duration = data.shape[1] / fs
    
    return freqs, t, Sxx, duration


def main():
    parser = argparse.ArgumentParser(
        description="Compute time-frequency map for a single EEG session"
    )
    
    # Session specification
    parser.add_argument(
        "--session-path",
        type=str,
        help="Direct path to session folder",
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        help="Index into session file (for Slurm array jobs)",
    )
    parser.add_argument(
        "--session-file",
        type=str,
        help="File containing session list (for Slurm array jobs)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Directory to save TF map results",
    )
    
    # Processing parameters
    parser.add_argument(
        "--target-fs", "-f",
        type=int,
        default=128,
        help="Target sampling frequency in Hz (default: 128)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers for loading (default: 4)",
    )
    parser.add_argument(
        "--contact-min",
        type=int,
        default=1,
        help="Minimum contact number to load (default: 1)",
    )
    parser.add_argument(
        "--contact-max",
        type=int,
        default=5,
        help="Maximum contact number to load (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if output exists",
    )
    
    args = parser.parse_args()
    
    # Determine session path
    session_path = None
    
    if args.session_path:
        session_path = args.session_path
    elif args.index is not None and args.session_file:
        # Read from session file (for Slurm array jobs)
        with open(args.session_file, 'r') as f:
            lines = f.readlines()
        if args.index < len(lines):
            session_path = lines[args.index].strip()
        else:
            print(f"Error: Index {args.index} out of range (file has {len(lines)} lines)")
            sys.exit(1)
    else:
        print("Error: Must specify --session-path or --index with --session-file")
        sys.exit(1)
    
    if not session_path or not os.path.isdir(session_path):
        print(f"Error: Session path does not exist: {session_path}")
        sys.exit(1)
    
    session_name = os.path.basename(session_path)
    print(f"Processing session: {session_name}")
    print(f"Session path: {session_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file
    output_file = output_dir / f"{session_name}_tf.npz"
    
    # Check if already exists
    if output_file.exists() and not args.force:
        print(f"Output already exists, skipping: {output_file}")
        print("Use --force to recompute")
        sys.exit(0)
    
    # Load EEG data
    print(f"Loading data (target_fs={args.target_fs} Hz)...")
    start_load = time.time()
    
    loader = FastNeuralynxLoader()
    try:
        data, channels, fs = loader.load_folder(
            session_path,
            target_fs=args.target_fs,
            max_workers=args.workers,
            contact_range=(args.contact_min, args.contact_max),
            parallel_method='thread'
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    if data.size == 0:
        print("Error: No data loaded")
        sys.exit(1)
    
    elapsed_load = time.time() - start_load
    print(f"Loaded in {elapsed_load:.1f}s | Shape: {data.shape} | Duration: {data.shape[1]/fs:.0f}s")
    
    # Compute TF map
    print("Computing time-frequency map...")
    start_tf = time.time()
    
    freqs, times, Sxx, duration = compute_tf_map(data, fs)
    
    elapsed_tf = time.time() - start_tf
    print(f"TF map computed in {elapsed_tf:.1f}s | {len(freqs)} freq bins x {len(times)} time bins")
    
    # Free memory
    del data
    
    # Save to disk
    print(f"Saving to: {output_file}")
    np.savez_compressed(
        output_file,
        session_name=session_name,
        freqs=freqs,
        times=times,
        Sxx=Sxx,
        duration=duration,
        fs=fs,
        n_channels=len(channels),
        channels=channels,
    )
    
    # Verify file
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Saved successfully ({file_size_mb:.2f} MB)")
    
    print("Done!")


if __name__ == "__main__":
    main()
