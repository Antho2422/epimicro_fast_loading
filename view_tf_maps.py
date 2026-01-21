"""
View precomputed Time-Frequency Maps from multiple sessions.

This script loads TF maps that were computed on the cluster and displays
them concatenated for quick overview and event spotting.

Usage:
    python view_tf_maps.py /path/to/tf_output_dir
    python view_tf_maps.py /path/to/tf_output_dir --pattern "03340*"
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
from fnmatch import fnmatch


def load_tf_maps(tf_dir, pattern="*_tf.npz"):
    """
    Load all precomputed TF maps from a directory.
    
    Parameters
    ----------
    tf_dir : str
        Directory containing .npz files
    pattern : str
        Glob pattern to filter files
    
    Returns
    -------
    session_data : list of dict
        Each dict contains session info and TF data
    """
    tf_path = Path(tf_dir)
    if not tf_path.exists():
        print(f"Error: Directory does not exist: {tf_dir}")
        return []
    
    # Find all .npz files
    npz_files = sorted(tf_path.glob(pattern))
    
    if not npz_files:
        print(f"No TF map files found matching pattern: {pattern}")
        return []
    
    print(f"Found {len(npz_files)} TF map files")
    
    session_data = []
    for npz_file in npz_files:
        print(f"  Loading: {npz_file.name}...", end=" ")
        try:
            data = np.load(npz_file, allow_pickle=True)
            session_data.append({
                'name': str(data['session_name']),
                'freqs': data['freqs'],
                'times': data['times'],
                'Sxx': data['Sxx'],
                'duration': float(data['duration']),
                'file': str(npz_file),
            })
            print(f"OK ({data['duration']:.0f}s)")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return session_data


class MultiSessionTFViewer:
    """Interactive viewer for concatenated time-frequency maps from multiple sessions."""
    
    def __init__(self, session_data):
        """
        Initialize the multi-session viewer.
        
        Parameters
        ----------
        session_data : list of dict
            Each dict contains:
            - 'name': session name
            - 'freqs': frequency array
            - 'times': time array (relative to session start)
            - 'Sxx': spectrogram (freq x time)
            - 'duration': session duration in seconds
        """
        self.session_data = session_data
        self.n_sessions = len(session_data)
        
        if self.n_sessions == 0:
            print("No session data to display!")
            return
        
        # Use frequencies from first session (should be same for all)
        self.freqs = session_data[0]['freqs']
        
        # Calculate cumulative time offsets for each session
        self.session_starts = []
        self.session_ends = []
        cumulative_time = 0
        
        for sess in session_data:
            self.session_starts.append(cumulative_time)
            cumulative_time += sess['duration']
            self.session_ends.append(cumulative_time)
        
        self.total_duration = cumulative_time
        
        # Concatenate all spectrograms
        self._concatenate_spectrograms()
        
        # View parameters
        self.window_duration = min(300, self.total_duration)  # 5 min default or total
        self.current_time = 0
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 8))
        
        # Main TF plot
        self.ax_tf = self.fig.add_axes([0.06, 0.25, 0.92, 0.65])
        
        # Overview plot (full recording)
        self.ax_overview = self.fig.add_axes([0.06, 0.08, 0.92, 0.12])
        
        # Slider
        ax_slider = self.fig.add_axes([0.06, 0.02, 0.88, 0.03])
        self.time_slider = Slider(
            ax_slider, 'Time (s)', 
            0, max(0, self.total_duration - self.window_duration),
            valinit=0, valstep=1
        )
        self.time_slider.on_changed(self.on_slider_change)
        
        # Window indicator on overview
        self.window_indicator = None
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial plot
        self._plot_overview()
        self._plot_main()
        self._print_controls()
    
    def _concatenate_spectrograms(self):
        """Concatenate all session spectrograms with proper time alignment."""
        all_Sxx = []
        all_times = []
        
        time_offset = 0
        for sess in self.session_data:
            all_Sxx.append(sess['Sxx'])
            all_times.append(sess['times'] + time_offset)
            time_offset += sess['duration']
        
        self.concat_Sxx = np.hstack(all_Sxx)
        self.concat_times = np.concatenate(all_times)
        
        # Convert to dB
        self.concat_Sxx_db = 10 * np.log10(self.concat_Sxx + 1e-10)
        
        # Auto-scale using percentiles
        self.vmin = np.percentile(self.concat_Sxx_db, 5)
        self.vmax = np.percentile(self.concat_Sxx_db, 95)
    
    def _plot_overview(self):
        """Plot the full concatenated spectrogram overview."""
        self.ax_overview.clear()
        
        self.ax_overview.imshow(
            self.concat_Sxx_db,
            aspect='auto',
            origin='lower',
            extent=[0, self.total_duration, self.freqs[0], self.freqs[-1]],
            cmap='viridis',
            interpolation='bilinear',
            vmin=self.vmin,
            vmax=self.vmax
        )
        
        # Draw session boundaries
        for i, (start, end) in enumerate(zip(self.session_starts, self.session_ends)):
            self.ax_overview.axvline(start, color='white', linewidth=0.5, alpha=0.5)
        
        # Update window indicator
        self._update_window_indicator()
        
        self.ax_overview.set_ylabel('Hz', fontsize=8)
        self.ax_overview.set_xlabel('Time (s)', fontsize=8)
        self.ax_overview.tick_params(labelsize=7)
        self.ax_overview.set_xlim([0, self.total_duration])
    
    def _plot_main(self):
        """Plot the zoomed time-frequency map."""
        self.ax_tf.clear()
        
        # Find time indices for current window
        t_start = self.current_time
        t_end = self.current_time + self.window_duration
        
        # Find corresponding indices in concat_times
        idx_start = np.searchsorted(self.concat_times, t_start)
        idx_end = np.searchsorted(self.concat_times, t_end)
        
        if idx_start >= idx_end:
            idx_end = min(idx_start + 1, len(self.concat_times))
        
        # Get data slice
        Sxx_slice = self.concat_Sxx_db[:, idx_start:idx_end]
        
        if Sxx_slice.size > 0:
            self.ax_tf.imshow(
                Sxx_slice,
                aspect='auto',
                origin='lower',
                extent=[t_start, t_end, self.freqs[0], self.freqs[-1]],
                cmap='viridis',
                interpolation='bilinear',
                vmin=self.vmin,
                vmax=self.vmax
            )
        
        # Draw session boundaries within view
        for i, (start, end, sess) in enumerate(zip(self.session_starts, self.session_ends, self.session_data)):
            if start >= t_start and start <= t_end:
                self.ax_tf.axvline(start, color='white', linewidth=1, alpha=0.8)
                # Add session name label
                self.ax_tf.text(start + 2, self.freqs[-1] - 5, sess['name'], 
                               color='white', fontsize=8, fontweight='bold',
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Find which session(s) we're viewing
        current_sessions = self._get_sessions_in_view(t_start, t_end)
        title = f"Time-Frequency Map | {t_start:.0f}s - {t_end:.0f}s | Sessions: {', '.join(current_sessions)}"
        self.ax_tf.set_title(title, fontsize=10)
        
        self.ax_tf.set_ylabel('Frequency (Hz)', fontsize=9)
        self.ax_tf.set_xlabel('Time (s)', fontsize=9)
        self.ax_tf.tick_params(labelsize=8)
        
        self._update_window_indicator()
        self.fig.canvas.draw_idle()
    
    def _get_sessions_in_view(self, t_start, t_end):
        """Get names of sessions visible in the current time window."""
        sessions_in_view = []
        for i, (start, end, sess) in enumerate(zip(self.session_starts, self.session_ends, self.session_data)):
            # Check if session overlaps with view window
            if start < t_end and end > t_start:
                sessions_in_view.append(sess['name'])
        return sessions_in_view
    
    def _update_window_indicator(self):
        """Update the window position indicator on the overview."""
        if self.window_indicator is not None:
            self.window_indicator.remove()
        self.window_indicator = self.ax_overview.axvspan(
            self.current_time,
            self.current_time + self.window_duration,
            color='red', alpha=0.3, linewidth=0
        )
    
    def _print_controls(self):
        """Print keyboard controls."""
        print("\n" + "="*80)
        print("MULTI-SESSION TIME-FREQUENCY VIEWER - CONTROLS")
        print("="*80)
        print("Navigation:")
        print("  → / ←          : Move forward/backward (30 seconds)")
        print("  Page Up/Down   : Move forward/backward (1 window)")
        print("  Home / End     : Go to start/end")
        print("  Mouse Scroll   : Scroll through time")
        print("  Click Overview : Jump to clicked position")
        print("")
        print("Zoom:")
        print("  + / -          : Zoom in/out (time)")
        print("")
        print("Session Navigation:")
        print("  n / p          : Jump to next/previous session")
        print("")
        print("Info:")
        print(f"  Total sessions: {self.n_sessions}")
        print(f"  Total duration: {self.total_duration:.0f} seconds ({self.total_duration/3600:.1f} hours)")
        print("="*80)
        print("\nSessions loaded:")
        for i, sess in enumerate(self.session_data):
            print(f"  {i+1}. {sess['name']} ({sess['duration']:.0f}s)")
        print("")
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'right':
            self.current_time = min(
                self.current_time + 30,
                self.total_duration - self.window_duration
            )
        elif event.key == 'left':
            self.current_time = max(self.current_time - 30, 0)
        elif event.key == 'pagedown':
            self.current_time = min(
                self.current_time + self.window_duration,
                self.total_duration - self.window_duration
            )
        elif event.key == 'pageup':
            self.current_time = max(self.current_time - self.window_duration, 0)
        elif event.key == 'home':
            self.current_time = 0
        elif event.key == 'end':
            self.current_time = max(0, self.total_duration - self.window_duration)
        elif event.key == '+' or event.key == '=':
            self.window_duration = max(30, self.window_duration / 1.5)
            self.time_slider.valmax = max(0, self.total_duration - self.window_duration)
        elif event.key == '-' or event.key == '_':
            self.window_duration = min(self.total_duration, self.window_duration * 1.5)
            self.time_slider.valmax = max(0, self.total_duration - self.window_duration)
            self.current_time = min(self.current_time, self.total_duration - self.window_duration)
        elif event.key == 'n':
            # Jump to next session
            for start in self.session_starts:
                if start > self.current_time + 1:
                    self.current_time = min(start, self.total_duration - self.window_duration)
                    break
        elif event.key == 'p':
            # Jump to previous session
            for start in reversed(self.session_starts):
                if start < self.current_time - 1:
                    self.current_time = max(start, 0)
                    break
        else:
            return
        
        self.time_slider.set_val(self.current_time)
        self._plot_main()
    
    def on_scroll(self, event):
        """Handle mouse scroll."""
        step = 10  # seconds
        if event.button == 'up':
            self.current_time = min(
                self.current_time + step,
                self.total_duration - self.window_duration
            )
        elif event.button == 'down':
            self.current_time = max(self.current_time - step, 0)
        
        self.time_slider.set_val(self.current_time)
        self._plot_main()
    
    def on_click(self, event):
        """Handle mouse click on overview to jump to position."""
        if event.inaxes == self.ax_overview:
            # Jump to clicked time
            clicked_time = event.xdata
            if clicked_time is not None:
                self.current_time = max(0, min(
                    clicked_time - self.window_duration / 2,
                    self.total_duration - self.window_duration
                ))
                self.time_slider.set_val(self.current_time)
                self._plot_main()
    
    def on_slider_change(self, val):
        """Handle slider change."""
        self.current_time = val
        self._plot_main()
    
    def show(self):
        """Display the viewer."""
        plt.show()


def check_job_status(tf_dir, session_file=None):
    """
    Check status of TF computation jobs.
    
    Parameters
    ----------
    tf_dir : str
        Directory where TF maps are saved
    session_file : str, optional
        File containing expected sessions
    """
    tf_path = Path(tf_dir)
    
    # Find completed files
    completed = list(tf_path.glob("*_tf.npz"))
    completed_names = {f.stem.replace('_tf', '') for f in completed}
    
    print(f"\n{'='*60}")
    print(f"TF Map Computation Status")
    print(f"{'='*60}")
    print(f"Output directory: {tf_dir}")
    print(f"Completed: {len(completed)} sessions")
    
    if session_file and os.path.exists(session_file):
        with open(session_file, 'r') as f:
            expected = [os.path.basename(line.strip()) for line in f]
        
        pending = [s for s in expected if s not in completed_names]
        
        print(f"Expected:  {len(expected)} sessions")
        print(f"Pending:   {len(pending)} sessions")
        
        if pending:
            print(f"\nPending sessions:")
            for s in pending[:10]:
                print(f"  - {s}")
            if len(pending) > 10:
                print(f"  ... and {len(pending) - 10} more")
    
    print(f"{'='*60}\n")
    
    return len(completed)


def main():
    parser = argparse.ArgumentParser(
        description="View precomputed Time-Frequency Maps from multiple sessions"
    )
    parser.add_argument(
        "tf_dir",
        help="Directory containing precomputed TF map .npz files"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="*_tf.npz",
        help="Pattern to filter TF map files (default: *_tf.npz)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only check job status, don't display viewer"
    )
    parser.add_argument(
        "--session-file",
        type=str,
        help="Session file to compare against (for status check)"
    )
    
    args = parser.parse_args()
    
    if args.status:
        check_job_status(args.tf_dir, args.session_file)
        return
    
    # Load TF maps
    print(f"\nLoading TF maps from: {args.tf_dir}")
    session_data = load_tf_maps(args.tf_dir, args.pattern)
    
    if not session_data:
        print("No sessions loaded!")
        return
    
    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(session_data)} sessions")
    total_duration = sum(s['duration'] for s in session_data)
    print(f"Total duration: {total_duration:.0f}s ({total_duration/3600:.1f} hours)")
    print(f"{'='*60}")
    
    # Launch viewer
    viewer = MultiSessionTFViewer(session_data)
    viewer.show()


if __name__ == "__main__":
    main()
