import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segfast
from config import SEGFAST_FILE_PATH
import os


def load_segfast_file(file_path):
    """Load the SegFast file."""
    return segfast.open(file_path, engine='segyio')  # for apple silicon
    # Use engine='memmap' for all other systems

def load_headers(segfast_file):
    """Load headers from the SegFast file."""
    df = segfast_file.load_headers(['INLINE_3D', 'CROSSLINE_3D'])
    df.columns = ['TRACE_SEQUENCE_FILE', 'INLINE_3D', 'CROSSLINE_3D']
    return df

def get_dimensions(df, segfast_file):
    """Determine the dimensions of the 3D record."""
    inline_min, inline_max = df['INLINE_3D'].min(), df['INLINE_3D'].max()
    crossline_min, crossline_max = df['CROSSLINE_3D'].min(), df['CROSSLINE_3D'].max()
    n_samples = segfast_file.n_samples
    return inline_min, inline_max, crossline_min, crossline_max, n_samples

def select_slice(df, slice_type, inline_min, inline_max, crossline_min, crossline_max):
    """Select a slice based on the specified type."""
    if slice_type == 'INLINE_3D':
        slice_value = (inline_min + inline_max) // 2
        traces_to_load = df[df['INLINE_3D'] == slice_value]['TRACE_SEQUENCE_FILE'].tolist()
    else:
        slice_value = (crossline_min + crossline_max) // 2
        traces_to_load = df[df['CROSSLINE_3D'] == slice_value]['TRACE_SEQUENCE_FILE'].tolist()
    return slice_value, traces_to_load

def load_traces(segfast_file, traces_to_load):
    """Load the selected traces."""
    return segfast_file.load_traces(traces_to_load, buffer=None)

def visualize_slice(seismic_slice, slice_type, slice_value):
    """Visualize the seismic slice."""
    plt.figure(figsize=(12, 8))
    plt.imshow(seismic_slice.T, cmap='seismic', aspect='auto')
    plt.colorbar(label='Amplitude')
    plt.title(f"Seismic Slice ({slice_type} = {slice_value})")
    plt.xlabel('Trace Number')
    plt.ylabel('Time/Depth (samples)')
    plt.tight_layout()
    plt.show()


def save_current_slice(seismic_slice, slice_type, slice_value):
    """Save the current seismic slice as a numpy array."""
    # Check if the temp directory exists, create it if it doesn't
    if not os.path.exists(''):
        os.makedirs('')

    # Create a filename with the current coordinates and mode
    filename = f"temp/seismic_slice_{slice_type}_{slice_value}.npy"

    # Save the numpy array
    np.save(filename, seismic_slice)
    print(f"Saved slice to {filename}")
def main():
    # Step 1: Load the SegFast file and headers
    segfast_file = load_segfast_file(SEGFAST_FILE_PATH)
    df = load_headers(segfast_file)

    # Step 2: Determine the dimensions of the 3D record
    inline_min, inline_max, crossline_min, crossline_max, n_samples = get_dimensions(df, segfast_file)
    print(f"3D Record Dimensions:")
    print(f"INLINE_3D: from {inline_min} to {inline_max}")
    print(f"CROSSLINE_3D: from {crossline_min} to {crossline_max}")
    print(f"Number of samples: {n_samples}")

    # Step 3: Select a slice (default is middle INLINE_3D)
    slice_type = 'INLINE_3D'  # Can be changed to 'CROSSLINE_3D'
    slice_value, traces_to_load = select_slice(df, slice_type, inline_min, inline_max, crossline_min, crossline_max)
    print(f"Selected slice {slice_type} = {slice_value}")

    # Step 4: Load the traces
    traces = load_traces(segfast_file, traces_to_load)

    # Step 5: Convert to numpy array
    seismic_slice = traces

    # Step 6: Visualize
    visualize_slice(seismic_slice, slice_type, slice_value)

if __name__ == "__main__":
    main()