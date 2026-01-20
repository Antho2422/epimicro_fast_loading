import argparse
import time
from core import FastNeuralynxLoader
from visualization import InteractiveEEGViewer

# Default data folder (can be overridden via command line argument)
DEFAULT_DATA_FOLDER = r"path\to\your\neuralynx\data"  # <-- Change this to your default path


def main(data_folder: str, target_fs: int = 128, max_workers: int = 8, 
         contact_range: tuple = (1, 5), window_duration: int = 30, montage: str = 'average'):
    """
    Load Neuralynx data and launch interactive viewer.
    
    Parameters
    ----------
    data_folder : str
        Path to Neuralynx data folder containing .ncs files
    target_fs : int
        Target sampling frequency for downsampling (default: 128 Hz)
    max_workers : int
        Number of parallel workers for loading (default: 8)
    contact_range : tuple
        Min and max contact numbers to load (default: (1, 5))
    window_duration : int
        Initial viewer window duration in seconds (default: 30)
    montage : str
        Montage type: 'raw', 'average', or 'bipolar' (default: 'average')
    """
    # Create loader
    loader = FastNeuralynxLoader()

    # Load data
    print("Loading data...")
    start = time.time()

    data, channels, fs = loader.load_folder(
        data_folder,
        target_fs=target_fs,
        max_workers=max_workers,
        contact_range=contact_range,
        parallel_method='thread'
    )

    elapsed = time.time() - start

    print(f"\nLoading completed in {elapsed:.2f} seconds")
    print(f"Data shape: {data.shape}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Duration: {data.shape[1]/fs:.2f} seconds\n")

    # Launch interactive viewer
    viewer = InteractiveEEGViewer(data, channels, fs, window_duration=window_duration, montage=montage)
    viewer.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Neuralynx EEG data loader and viewer"
    )
    parser.add_argument(
        "data_folder",
        nargs="?",
        default=DEFAULT_DATA_FOLDER,
        help="Path to Neuralynx data folder (default: uses DEFAULT_DATA_FOLDER)"
    )
    parser.add_argument(
        "--target-fs", "-f",
        type=int,
        default=128,
        help="Target sampling frequency in Hz (default: 128)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--contact-min",
        type=int,
        default=1,
        help="Minimum contact number to load (default: 1)"
    )
    parser.add_argument(
        "--contact-max",
        type=int,
        default=5,
        help="Maximum contact number to load (default: 5)"
    )
    parser.add_argument(
        "--window", "-d",
        type=int,
        default=30,
        help="Initial viewer window duration in seconds (default: 30s)"
    )
    parser.add_argument(
        "--montage", "-m",
        type=str,
        default="average",
        choices=["raw", "average", "bipolar"],
        help="Montage type: raw, average, or bipolar (default: average)"
    )

    args = parser.parse_args()

    main(
        data_folder=args.data_folder,
        target_fs=args.target_fs,
        max_workers=args.workers,
        contact_range=(args.contact_min, args.contact_max),
        window_duration=args.window,
        montage=args.montage
    )