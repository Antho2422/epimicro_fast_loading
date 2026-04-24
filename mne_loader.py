"""
MNE-compatible loader for generic Neuralynx .ncs recordings.

Loads all .ncs files from a folder using the fast binary reading
process and returns a downsampled MNE RawArray.  Avoids the OOM
issues that arise when letting MNE load full-resolution data directly.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import mne
import numpy as np

from core import FastNeuralynxLoader


def load_ncs_folder_as_raw(
    folder_path: str,
    target_fs: int = 256,
    max_workers: Optional[int] = None,
    channel_type: Union[str, List[str]] = 'seeg',
) -> mne.io.RawArray:
    """
    Load all .ncs files from a folder and return a downsampled MNE Raw object.

    Reads each file in parallel using the fast binary loading approach
    (direct struct parsing, no neo/MNE overhead), downsamples to
    ``target_fs``, then wraps the result in an MNE RawArray.

    Channel names are taken directly from the file stems (e.g.
    ``CSC1.ncs`` → ``"CSC1"``).  No channel filtering is applied —
    every .ncs file in the folder is loaded.

    Parameters
    ----------
    folder_path : str
        Path to folder containing .ncs files.
    target_fs : int
        Target sampling frequency in Hz (default: 256 Hz).
    max_workers : int, optional
        Number of parallel loading threads.  ``None`` lets the
        ThreadPoolExecutor choose based on the number of CPUs.
    channel_type : str or list of str
        MNE channel type(s).  Either a single string applied to all
        channels, or a list of the same length as the number of
        channels found (default: ``'seeg'``).

    Returns
    -------
    mne.io.RawArray
        MNE Raw object with all channels in Volts at ``target_fs``.

    Raises
    ------
    ValueError
        If no .ncs files are found or none could be loaded.

    Examples
    --------
    >>> raw = load_ncs_folder_as_raw('/data/session/', target_fs=512)
    >>> raw.plot()
    """
    loader = FastNeuralynxLoader()

    ncs_files = sorted(glob.glob(os.path.join(folder_path, '*.ncs')))
    if not ncs_files:
        raise ValueError(f"No .ncs files found in {folder_path}")

    print(
        f"Found {len(ncs_files)} .ncs file(s) in "
        f"{os.path.basename(os.path.normpath(folder_path))}"
    )

    channel_names: List[str] = []
    signals: List[np.ndarray] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(loader.read_ncs_file, f, target_fs): f
            for f in ncs_files
        }

        for i, future in enumerate(futures):
            filepath = futures[future]
            try:
                data, _orig_fs, _final_fs = future.result()
                if len(data) > 0:
                    channel_names.append(Path(filepath).stem)
                    signals.append(data)
            except Exception as exc:
                print(f"  Warning: could not load {filepath}: {exc}")

            if (i + 1) % 10 == 0 or (i + 1) == len(ncs_files):
                print(f"  Loaded {i + 1}/{len(ncs_files)} file(s)...")

    if not signals:
        raise ValueError("No data could be loaded from the provided folder.")

    # Pad channels to the same length (last record may differ slightly)
    max_len = max(len(s) for s in signals)
    data_matrix = np.zeros((len(signals), max_len), dtype=np.float32)
    for i, sig in enumerate(signals):
        data_matrix[i, :len(sig)] = sig

    # read_ncs_file returns µV; MNE expects Volts
    data_matrix *= 1e-6

    _validate_channel_type(channel_type, channel_names)

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=float(target_fs),
        ch_types=channel_type,
    )
    raw = mne.io.RawArray(data_matrix, info, verbose=False)

    n_ch = len(channel_names)
    duration = data_matrix.shape[1] / target_fs
    print(
        f"\nReady: {n_ch} channel(s), {target_fs} Hz, "
        f"{duration:.2f} s ({duration / 60:.1f} min)"
    )

    return raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_channel_type(
    channel_type: Union[str, List[str]],
    channel_names: List[str],
) -> None:
    """Raise ValueError if channel_type length mismatches channel count."""
    if isinstance(channel_type, list):
        if len(channel_type) != len(channel_names):
            raise ValueError(
                f"channel_type list length ({len(channel_type)}) must match "
                f"the number of loaded channels ({len(channel_names)})."
            )
