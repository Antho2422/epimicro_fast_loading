# EpiMicro Fast Loading

A fast Neuralynx (.ncs) file loader optimized for EEG/iEEG data visualization.

## Features

- **Fast parallel loading** of Neuralynx .ncs files using multithreading or multiprocessing
- **Automatic downsampling** for efficient visualization
- **Smart channel filtering** - loads only macroelectrode contacts within specified ranges
- **Interactive EEG viewer** with keyboard navigation and zoom controls

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python main.py "path/to/neuralynx/folder"
```

### In Code

```python
from core import FastNeuralynxLoader
from visualization import InteractiveEEGViewer

loader = FastNeuralynxLoader()
data, channels, fs = loader.load_folder(
    "path/to/data",
    target_fs=128,
    contact_range=(1, 9)
)

viewer = InteractiveEEGViewer(data, channels, fs)
viewer.show()
```

## Viewer Controls

| Key | Action |
|-----|--------|
| ← / → | Move backward/forward 1 second |
| Page Up/Down | Move 1 window |
| Home / End | Go to start/end |
| + / - | Zoom in/out (time) |
| ↑ / ↓ | Increase/decrease channel spacing |
| Mouse Scroll | Scroll through time |