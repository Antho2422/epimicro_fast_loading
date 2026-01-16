import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class InteractiveEEGViewer:
    """Interactive EEG viewer with scroll and zoom capabilities"""
    
    def __init__(self, data, channel_names, fs, window_duration=10):
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
        """
        self.data = data
        self.channel_names = channel_names
        self.fs = fs
        self.n_channels, self.n_samples = data.shape
        self.total_duration = self.n_samples / fs
        
        # View parameters
        self.window_duration = window_duration
        self.current_time = 0
        self.offset_scale = 5.0
        self.show_all_channels = True
        self.selected_channels = list(range(self.n_channels))
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main plot area
        self.ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        
        # Connect keyboard and scroll events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Create slider for time navigation
        ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.time_slider = Slider(
            ax_slider, 'Time (s)', 
            0, max(0, self.total_duration - self.window_duration),
            valinit=0, valstep=0.1
        )
        self.time_slider.on_changed(self.on_slider_change)
        
        # Initial plot
        self.plot()
        
        # Print controls
        self.print_controls()
    
    def print_controls(self):
        """Print keyboard controls"""
        print("\n" + "="*80)
        print("INTERACTIVE EEG VIEWER - CONTROLS")
        print("="*80)
        print("Navigation:")
        print("  → / ←          : Move forward/backward (1 second)")
        print("  Page Up/Down   : Move forward/backward (1 window)")
        print("  Home / End     : Go to start/end")
        print("  Mouse Scroll   : Scroll through time")
        print("")
        print("Zoom:")
        print("  + / -          : Zoom in/out (time)")
        print("  ↑ / ↓          : Increase/decrease channel spacing")
        print("")
        print("Display:")
        print("  a              : Toggle auto-scale")
        print("  r              : Reset view")
        print("  g              : Toggle grid")
        print("")
        print("Info:")
        print(f"  Total duration: {self.total_duration:.2f} seconds")
        print(f"  Channels: {self.n_channels}")
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
            
            # Normalize
            if signal_data.std() > 0:
                signal_data = signal_data / signal_data.std()
            
            self.ax.plot(time, signal_data + offsets[i], 
                        linewidth=0.5, color='black', alpha=0.8)
        
        # Labels and formatting
        self.ax.set_xlabel('Time (s)', fontsize=12)
        self.ax.set_ylabel('Channel', fontsize=12)
        
        title = f'EEG Viewer - {n_plot_channels} channels @ {self.fs} Hz | '
        title += f'Window: {self.current_time:.1f} - {self.current_time + self.window_duration:.1f} s'
        self.ax.set_title(title, fontsize=12)
        
        # Set y-ticks
        plot_channel_names = [self.channel_names[i] for i in self.selected_channels]
        self.ax.set_yticks(offsets)
        self.ax.set_yticklabels(plot_channel_names, fontsize=8)
        
        self.ax.set_xlim([self.current_time, self.current_time + self.window_duration])
        self.ax.grid(True, alpha=0.3, axis='x')
        
        self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            # Move forward 1 second
            self.current_time = min(
                self.current_time + 1, 
                self.total_duration - self.window_duration
            )
        
        elif event.key == 'left':
            # Move backward 1 second
            self.current_time = max(self.current_time - 1, 0)
        
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
        
        elif event.key == 'r':
            # Reset view
            self.current_time = 0
            self.window_duration = 10
            self.offset_scale = 5.0
        
        elif event.key == 'g':
            # Toggle grid
            self.ax.grid(not self.ax.xaxis._gridOnMajor)
        
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
