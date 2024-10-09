import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons, Button
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


def load_traces(segfast_file, traces_to_load):
    """Load the selected traces, skipping problematic ones."""
    loaded_traces = []
    for trace in traces_to_load:
        try:
            loaded_trace = segfast_file.load_traces([trace], buffer=None)
            loaded_traces.append(loaded_trace[0])
        except OSError:
            print(f"Warning: Failed to load trace {trace}. Skipping.")
    return np.array(loaded_traces)


def update_slice(slice_type, slice_value, df, segfast_file):
    """Update the seismic slice based on the selected slice type and value."""
    if slice_type == 'DEPTH':
        depth_slice = segfast_file.load_depth_slices([slice_value], buffer=None)[0]
        return depth_slice.reshape(df['INLINE_3D'].nunique(), df['CROSSLINE_3D'].nunique())
    else:
        traces_to_load = df[df[slice_type] == slice_value]['TRACE_SEQUENCE_FILE'].tolist()
        traces = load_traces(segfast_file, traces_to_load)
        return traces

def visualize_slice(seismic_slice, slice_type, slice_value, ax, im, cbar, title, cmap, vmin, vmax):
    """Visualize the seismic slice."""
    im.set_data(seismic_slice.T if slice_type != 'DEPTH' else seismic_slice)
    im.set_cmap(cmap)
    im.set_clim(vmin=vmin, vmax=vmax)
    title.set_text(f"Seismic Slice ({slice_type} = {slice_value})")
    if slice_type == 'DEPTH':
        ax.set_xlabel('INLINE_3D')
        ax.set_ylabel('CROSSLINE_3D')
    elif slice_type == 'INLINE_3D':
        ax.set_xlabel('CROSSLINE_3D')
        ax.set_ylabel('Time/Depth (samples)')
    else:  # CROSSLINE_3D
        ax.set_xlabel('INLINE_3D')
        ax.set_ylabel('Time/Depth (samples)')
    plt.draw()


def get_global_min_max(segfast_file, df):
    """Get global min and max amplitude values using batched processing."""
    print("Calculating global min and max values. This may take a while...")
    global_min = float('inf')
    global_max = float('-inf')
    batch_size = 1000  # Adjust this value based on your system's memory capacity

    for i in range(0, len(df), batch_size):
        batch_traces = df['TRACE_SEQUENCE_FILE'].iloc[i:i + batch_size].tolist()
        batch_data = load_traces(segfast_file, batch_traces)
        if len(batch_data) > 0:
            batch_min = np.min(batch_data)
            batch_max = np.max(batch_data)
            global_min = min(global_min, batch_min)
            global_max = max(global_max, batch_max)
        print(f"Processed {i + len(batch_traces)} out of {len(df)} traces.")

    if global_min == float('inf') or global_max == float('-inf'):
        raise ValueError("Failed to calculate global min and max values. No valid data found.")

    print(f"Global min: {global_min}, Global max: {global_max}")
    return global_min, global_max


def save_current_slice(seismic_slice, slice_type, slice_value):
    """Save the current seismic slice as a numpy array."""
    if not os.path.exists(''):
        os.makedirs('')

    filename = f"temp/seismic_slice_{slice_type}_{slice_value}.npy"

    np.save(filename, seismic_slice)
    print(f"Saved slice to {filename}")


def main():
    # Load data
    segfast_file = load_segfast_file(SEGFAST_FILE_PATH)
    df = load_headers(segfast_file)
    inline_min, inline_max, crossline_min, crossline_max, n_samples = get_dimensions(df, segfast_file)

    # Get global min and max values
    try:
        global_min, global_max = get_global_min_max(segfast_file, df)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create the main figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Initial slice
    initial_slice_type = 'INLINE_3D'
    initial_slice_value = (inline_min + inline_max) // 2
    seismic_slice = update_slice(initial_slice_type, initial_slice_value, df, segfast_file)

    # Create the initial image
    cmap = 'seismic'
    im = ax.imshow(seismic_slice.T, cmap=cmap, aspect='auto', vmin=global_min, vmax=global_max)
    cbar = plt.colorbar(im, ax=ax, label='Amplitude')
    title = ax.set_title(f"Seismic Slice ({initial_slice_type} = {initial_slice_value})")

    # Create slider
    slider_ax = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(slider_ax, 'INLINE_3D', inline_min, inline_max, valinit=initial_slice_value, valstep=1)

    # Create text box
    text_box_ax = plt.axes([0.1, 0.05, 0.2, 0.03])
    text_box = TextBox(text_box_ax, 'Slice Value', initial=str(initial_slice_value))

    # Create radio buttons for slice type
    rax = plt.axes([0.35, 0.05, 0.3, 0.05])
    radio = RadioButtons(rax, ('INLINE_3D', 'CROSSLINE_3D', 'DEPTH'))

    # Create button for color scheme switching
    color_button_ax = plt.axes([0.66, 0.05, 0.1, 0.05])
    color_button = Button(color_button_ax, 'Switch Color')

    # Create button for saving the current slice
    save_button_ax = plt.axes([0.77, 0.05, 0.1, 0.05])
    save_button = Button(save_button_ax, 'Save Slice')

    def update(val):
        slice_type = radio.value_selected
        slice_value = int(slider.val)
        text_box.set_val(str(slice_value))
        seismic_slice = update_slice(slice_type, slice_value, df, segfast_file)
        visualize_slice(seismic_slice, slice_type, slice_value, ax, im, cbar, title, cmap, global_min, global_max)
        return seismic_slice

    def update_from_text(text):
        try:
            val = int(text)
            slider.set_val(val)
        except ValueError:
            pass

    def update_slider_range(label):
        if label == 'INLINE_3D':
            slider.valmin = inline_min
            slider.valmax = inline_max
            slider.label.set_text('INLINE_3D')
            axis_size = df['INLINE_3D'].nunique()
        elif label == 'CROSSLINE_3D':
            slider.valmin = crossline_min
            slider.valmax = crossline_max
            slider.label.set_text('CROSSLINE_3D')
            axis_size = df['CROSSLINE_3D'].nunique()
        else:  # DEPTH
            slider.valmin = 0
            slider.valmax = n_samples - 1
            slider.label.set_text('DEPTH')
            axis_size = n_samples

        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.valstep = max(1, (slider.valmax - slider.valmin) // axis_size)
        slider.set_val((slider.valmin + slider.valmax) // 2)
        text_box.set_val(str((slider.valmin + slider.valmax) // 2))
        update(slider.val)

    def switch_color(event):
        nonlocal cmap
        cmap = 'gray' if cmap == 'seismic' else 'seismic'
        update(slider.val)

    def save_current(event):
        slice_type = radio.value_selected
        slice_value = int(slider.val)
        seismic_slice = update(slice_value)
        save_current_slice(seismic_slice, slice_type, slice_value)

    slider.on_changed(update)
    text_box.on_submit(update_from_text)
    radio.on_clicked(update_slider_range)
    color_button.on_clicked(switch_color)
    save_button.on_clicked(save_current)

    # Add text to show current range and axis size
    range_text = plt.text(0.1, 0.01, f"Range: {inline_min} - {inline_max}, Axis Size: {df['INLINE_3D'].nunique()}",
                          transform=fig.transFigure)

    def update_range_text(label):
        if label == 'INLINE_3D':
            range_text.set_text(f"Range: {inline_min} - {inline_max}, Axis Size: {df['INLINE_3D'].nunique()}")
        elif label == 'CROSSLINE_3D':
            range_text.set_text(f"Range: {crossline_min} - {crossline_max}, Axis Size: {df['CROSSLINE_3D'].nunique()}")
        else:  # DEPTH
            range_text.set_text(f"Range: 0 - {n_samples - 1}, Axis Size: {n_samples}")
        plt.draw()

    radio.on_clicked(update_range_text)

    plt.show()


if __name__ == "__main__":
    main()