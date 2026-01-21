import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import spectrogram, decimate

class InteractiveEEGViewer:
    """Interactive EEG viewer with scroll and zoom capabilities"""
    
    def __init__(self, data, channel_names, fs, window_duration=30, montage='raw', session_name=None):
        """
        Initialize interactive viewer
        
        Parameters:
        -----------
        data : array (n_channels, n_samples)
        channel_names : list of str
        fs : float
            Sampling frequency
        window_duration : float
            Initial window duration in seconds
        montage : str
            Montage type: 'raw', 'average', or 'bipolar'
        session_name : str, optional
            Name of the session to display in the title
        """
        self.session_name = session_name if session_name else ''
        self.raw_data = data  # Keep original data
        self.original_channel_names = list(channel_names)  # Keep original channel names
        self.channel_names = list(channel_names)  # Current display names (may change with montage)
        self.fs = fs
        self.n_channels, self.n_samples = data.shape
        self.total_duration = self.n_samples / fs
        
        # Montage settings
        self.montage = montage
        self.data = self._apply_montage(data, montage)
        
        # View parameters
        self.window_duration = window_duration
        self.current_time = 0
        self.offset_scale = 5.0
        self.gain = 1.0  # Signal amplitude gain
        self.show_all_channels = True
        self.selected_channels = list(range(self.data.shape[0]))
        
        # Create figure - compact size
        self.fig = plt.figure(figsize=(14, 8))
        
        # Main plot area - maximize data display space (leave room for TF map at bottom)
        self.ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.15, left=0.08, right=0.98, top=0.95)
        
        # Connect keyboard and scroll events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Create slider for time navigation - moved up to make room for TF map
        ax_slider = plt.axes([0.08, 0.10, 0.88, 0.02])
        self.time_slider = Slider(
            ax_slider, 'Time (s)', 
            0, max(0, self.total_duration - self.window_duration),
            valinit=0, valstep=0.1
        )
        self.time_slider.on_changed(self.on_slider_change)
        
        # Create time-frequency overview axes (below slider)
        self.ax_tf = self.fig.add_axes([0.08, 0.02, 0.88, 0.06])
        self.window_indicator = None
        
        # Precompute and plot time-frequency overview
        self._compute_overview_spectrogram()
        self._plot_overview_spectrogram()
        
        # Initial plot
        self.plot()
        
        # Print controls
        self.print_controls()
    
    def _apply_montage(self, data, montage):
        """
        Apply montage to data
        
        Parameters:
        -----------
        data : array (n_channels, n_samples)
        montage : str
            'raw': no re-referencing
            'average': subtract average of all channels
            'bipolar': subtract adjacent channels (same electrode)
        
        Returns:
        --------
        data : array - re-referenced data
        """
        if montage == 'raw':
            return data
        
        elif montage == 'average':
            # Average reference: subtract mean of all channels at each time point
            avg = np.mean(data, axis=0, keepdims=True)
            return data - avg
        
        elif montage == 'bipolar':
            # Bipolar: subtract adjacent contacts on same electrode
            # Group channels by electrode name
            return self._compute_bipolar(data)
        
        else:
            print(f"Unknown montage '{montage}', using raw")
            return data
    
    def _compute_bipolar(self, data):
        """Compute bipolar montage (adjacent contact subtraction)
        
        For each electrode, compute the difference between adjacent contacts.
        The last contact of each electrode is removed (no pair to subtract).
        """
        from collections import defaultdict
        
        # Group channels by electrode name using ORIGINAL channel names
        electrode_groups = defaultdict(list)
        for i, name in enumerate(self.original_channel_names):
            if '_' in name:
                electrode, contact = name.rsplit('_', 1)
                try:
                    contact_num = int(contact)
                    electrode_groups[electrode].append((contact_num, i, name))
                except ValueError:
                    pass
        
        # Sort each group by contact number and compute bipolar pairs
        bipolar_data = []
        bipolar_names = []
        
        for electrode, contacts in sorted(electrode_groups.items()):
            contacts.sort(key=lambda x: x[0])  # Sort by contact number
            # Create pairs: contact1-contact2, contact2-contact3, etc.
            # Last contact is excluded (no next contact to subtract)
            for j in range(len(contacts) - 1):
                contact1_num, idx1, name1 = contacts[j]
                contact2_num, idx2, name2 = contacts[j + 1]
                
                # Bipolar = contact_i - contact_(i+1)
                bipolar_data.append(data[idx1] - data[idx2])
                bipolar_names.append(f"{electrode}_{contact1_num}-{contact2_num}")
        
        if bipolar_data:
            self.channel_names = bipolar_names
            return np.array(bipolar_data)
        else:
            print("Could not compute bipolar montage, using raw")
            self.channel_names = list(self.original_channel_names)
            return data
    
    def set_montage(self, montage):
        """Change montage and refresh display"""
        # Restore original channel names before applying new montage
        self.channel_names = list(self.original_channel_names)
        self.montage = montage
        self.data = self._apply_montage(self.raw_data, montage)
        self.selected_channels = list(range(self.data.shape[0]))
        print(f"Montage changed to: {montage}")
        self.plot()
    
    def _compute_overview_spectrogram(self):
        """Precompute time-frequency map for overview display (average of all channels, 1-100 Hz)"""
        print("Computing time-frequency overview...")
        
        # Average all channels for overview
        overview_signal = np.mean(self.raw_data, axis=0)
        
        # Downsample if fs > 512 Hz for faster computation
        effective_fs = self.fs
        if self.fs > 512:
            factor = int(self.fs // 256)
            overview_signal = decimate(overview_signal, factor)
            effective_fs = self.fs / factor
        
        # Compute spectrogram with coarse time resolution for overview
        # Use 2s windows with 50% overlap for a good balance of speed and resolution
        nperseg = int(effective_fs * 2)
        noverlap = int(effective_fs * 1)
        
        f, t, Sxx = spectrogram(
            overview_signal,
            fs=effective_fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=max(256, nperseg),
        )
        
        # Keep only 1-100 Hz (EEG-relevant frequencies)
        freq_mask = (f >= 1) & (f <= 100)
        self.overview_freqs = f[freq_mask]
        self.overview_times = t
        self.overview_Sxx = Sxx[freq_mask, :]
        
        print(f"Time-frequency overview computed: {len(self.overview_freqs)} freq bins, {len(self.overview_times)} time bins")
    
    def _plot_overview_spectrogram(self):
        """Display the precomputed spectrogram overview with auto-scaled colormap"""
        self.ax_tf.clear()
        
        # Convert to dB scale for better visualization
        Sxx_db = 10 * np.log10(self.overview_Sxx + 1e-10)
        
        # Auto-scale: use percentiles to avoid outliers dominating the colormap
        vmin = np.percentile(Sxx_db, 5)
        vmax = np.percentile(Sxx_db, 95)
        
        self.tf_image = self.ax_tf.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=[0, self.total_duration, 
                    self.overview_freqs[0], self.overview_freqs[-1]],
            cmap='viridis',
            interpolation='bilinear',
            vmin=vmin,
            vmax=vmax
        )
        
        # Add current window indicator
        self._update_window_indicator()
        
        self.ax_tf.set_ylabel('Hz', fontsize=7)
        self.ax_tf.set_xlim([0, self.total_duration])
        self.ax_tf.tick_params(labelsize=6)
        # Remove x-axis labels (redundant with slider)
        self.ax_tf.set_xticklabels([])
    
    def _update_window_indicator(self):
        """Update the position indicator on the time-frequency map"""
        if self.window_indicator is not None:
            self.window_indicator.remove()
        self.window_indicator = self.ax_tf.axvspan(
            self.current_time, 
            self.current_time + self.window_duration,
            color='red', alpha=0.3, linewidth=0
        )
    
    def print_controls(self):
        """Print keyboard controls"""
        print("\n" + "="*80)
        print("INTERACTIVE EEG VIEWER - CONTROLS")
        print("="*80)
        print("Navigation:")
        print("  → / ←          : Move forward/backward (5 seconds)")
        print("  Page Up/Down   : Move forward/backward (1 window)")
        print("  Home / End     : Go to start/end")
        print("  Mouse Scroll   : Scroll through time")
        print("")
        print("Zoom:")
        print("  + / -          : Zoom in/out (time)")
        print("  ↑ / ↓          : Increase/decrease channel spacing")
        print("  ] / [          : Increase/decrease signal gain")
        print("")
        print("Display:")
        print("  a              : Toggle auto-scale")
        print("  r              : Reset view")
        print("  g              : Toggle grid")
        print("")
        print("Montage:")
        print("  m              : Cycle montage (raw -> average -> bipolar)")
        print(f"  Current: {self.montage}")
        print("")
        print("Info:")
        print(f"  Total duration: {self.total_duration:.2f} seconds")
        print(f"  Channels: {self.data.shape[0]}")
        print(f"  Sampling rate: {self.fs} Hz")
        print("="*80 + "\n")
    
    def plot(self):
        """Plot current window"""
        self.ax.clear()
        
        # Calculate sample range
        start_sample = int(self.current_time * self.fs)
        end_sample = int((self.current_time + self.window_duration) * self.fs)
        
        # Clip to valid range
        start_sample = max(0, min(start_sample, self.n_samples))
        end_sample = max(0, min(end_sample, self.n_samples))
        
        n_display_samples = end_sample - start_sample
        
        if n_display_samples <= 0:
            return
        
        # Time vector
        time = np.arange(n_display_samples) / self.fs + self.current_time
        
        # Get data to plot
        data_to_plot = self.data[self.selected_channels, start_sample:end_sample]
        
        # Calculate offsets
        n_plot_channels = len(self.selected_channels)
        offsets = np.arange(n_plot_channels)[::-1] * self.offset_scale
        
        # Plot each channel
        for i, ch_idx in enumerate(self.selected_channels):
            signal_data = data_to_plot[i]
            
            # Normalize and apply gain
            if signal_data.std() > 0:
                signal_data = signal_data / signal_data.std() * self.gain
            
            self.ax.plot(time, signal_data + offsets[i], 
                        linewidth=0.5, color='black', alpha=0.8)
        
        # Labels and formatting - compact styling
        self.ax.set_xlabel('Time (s)', fontsize=9)
        self.ax.set_ylabel('Channel', fontsize=9)
        
        # Session name as main title (suptitle)
        if self.session_name:
            self.fig.suptitle(self.session_name, fontsize=11, fontweight='bold')
        
        # Plot info as subtitle
        subtitle = f'[{self.montage.upper()}] {n_plot_channels} ch @ {self.fs} Hz | '
        subtitle += f'{self.current_time:.1f}-{self.current_time + self.window_duration:.1f}s | Gain: {self.gain:.1f}x'
        self.ax.set_title(subtitle, fontsize=9)
        
        # Set y-ticks - smaller font for channel names
        plot_channel_names = [self.channel_names[i] for i in self.selected_channels]
        self.ax.set_yticks(offsets)
        self.ax.set_yticklabels(plot_channel_names, fontsize=7)
        
        self.ax.set_xlim([self.current_time, self.current_time + self.window_duration])
        self.ax.grid(True, alpha=0.3, axis='x')
        
        # Update window indicator on time-frequency map
        self._update_window_indicator()
        
        self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            # Move forward 5 seconds
            self.current_time = min(
                self.current_time + 5, 
                self.total_duration - self.window_duration
            )
        
        elif event.key == 'left':
            # Move backward 5 seconds
            self.current_time = max(self.current_time - 5, 0)
        
        elif event.key == 'pagedown':
            # Move forward 1 window
            self.current_time = min(
                self.current_time + self.window_duration, 
                self.total_duration - self.window_duration
            )
        
        elif event.key == 'pageup':
            # Move backward 1 window
            self.current_time = max(
                self.current_time - self.window_duration, 0
            )
        
        elif event.key == 'home':
            # Go to start
            self.current_time = 0
        
        elif event.key == 'end':
            # Go to end
            self.current_time = max(0, self.total_duration - self.window_duration)
        
        elif event.key == '+' or event.key == '=':
            # Zoom in (decrease window)
            self.window_duration = max(1, self.window_duration / 1.5)
            self.time_slider.valmax = max(0, self.total_duration - self.window_duration)
        
        elif event.key == '-' or event.key == '_':
            # Zoom out (increase window)
            self.window_duration = min(self.total_duration, self.window_duration * 1.5)
            self.time_slider.valmax = max(0, self.total_duration - self.window_duration)
            self.current_time = min(self.current_time, self.total_duration - self.window_duration)
        
        elif event.key == 'up':
            # Increase channel spacing
            self.offset_scale *= 1.2
        
        elif event.key == 'down':
            # Decrease channel spacing
            self.offset_scale /= 1.2
        
        elif event.key == ']':
            # Increase gain
            self.gain *= 1.5
            print(f"Gain: {self.gain:.2f}x")
        
        elif event.key == '[':
            # Decrease gain
            self.gain /= 1.5
            print(f"Gain: {self.gain:.2f}x")
        
        elif event.key == 'r':
            # Reset view
            self.current_time = 0
            self.window_duration = 10
            self.offset_scale = 5.0
            self.gain = 1.0
        
        elif event.key == 'g':
            # Toggle grid
            self.ax.grid(not self.ax.xaxis._gridOnMajor)
        
        elif event.key == 'm':
            # Cycle montage: raw -> average -> bipolar -> raw
            montages = ['raw', 'average', 'bipolar']
            current_idx = montages.index(self.montage) if self.montage in montages else 0
            next_idx = (current_idx + 1) % len(montages)
            self.set_montage(montages[next_idx])
            return  # set_montage already calls plot()
        
        else:
            return
        
        # Update slider
        self.time_slider.set_val(self.current_time)
        self.plot()
    
    def on_scroll(self, event):
        """Handle mouse scroll events"""
        # Scroll up = forward, scroll down = backward
        step = 0.5  # seconds
        
        if event.button == 'up':
            self.current_time = min(
                self.current_time + step, 
                self.total_duration - self.window_duration
            )
        elif event.button == 'down':
            self.current_time = max(self.current_time - step, 0)
        
        self.time_slider.set_val(self.current_time)
        self.plot()
    
    def on_slider_change(self, val):
        """Handle slider change"""
        self.current_time = val
        self.plot()
    
    def show(self):
        """Show the viewer"""
        plt.show()


# Enhanced version with channel selection
class AdvancedEEGViewer(InteractiveEEGViewer):
    """Advanced viewer with channel selection and filtering"""
    
    def __init__(self, data, channel_names, fs, window_duration=10):
        # Add channel filtering
        self.all_channel_indices = list(range(data.shape[0]))
        self.filter_text = ""
        
        super().__init__(data, channel_names, fs, window_duration)
    
    def print_controls(self):
        """Print keyboard controls"""
        super().print_controls()
        print("Advanced controls:")
        print("  f              : Filter channels (type electrode name)")
        print("  c              : Clear channel filter")
        print("  1-9            : Show only every Nth channel")
        print("="*80 + "\n")
    
    def on_key(self, event):
        """Enhanced keyboard handler"""
        
        if event.key == 'c':
            # Clear filter
            self.selected_channels = self.all_channel_indices
            self.filter_text = ""
            print(f"Filter cleared. Showing all {len(self.selected_channels)} channels")
            self.plot()
            return
        
        elif event.key in '123456789':
            # Show every Nth channel
            n = int(event.key)
            self.selected_channels = self.all_channel_indices[::n]
            print(f"Showing every {n}th channel ({len(self.selected_channels)} channels)")
            self.plot()
            return
        
        # Call parent handler
        super().on_key(event)
    
    def filter_channels(self, search_text):
        """Filter channels by name"""
        self.filter_text = search_text.lower()
        self.selected_channels = [
            i for i in self.all_channel_indices 
            if self.filter_text in self.channel_names[i].lower()
        ]
        print(f"Filter '{search_text}': showing {len(self.selected_channels)} channels")
        self.plot()
