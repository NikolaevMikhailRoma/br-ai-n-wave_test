import pandas as pd
import segfast
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons, Button
from config import SEGFAST_FILE_PATH
import os
import logging
from datetime import datetime


# Создаем директорию для логов, если она не существует
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Настройка логирования
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(log_dir, f"seismic_reader_log_{current_time}.log")

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Это позволит выводить логи и в консоль, и в файл
    ]
)

class SeismicSliceReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            self.segfast_file = self._load_segfast_file()
            self.df = self._load_headers()
            self.dimensions = self._get_dimensions()
            logging.info(f"Successfully initialized SeismicSliceReader with file: {file_path}")
            logging.info(f"Dimensions: {self.dimensions}")
        except Exception as e:
            logging.error(f"Error initializing SeismicSliceReader: {e}")
            raise

    def _load_segfast_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        logging.info(f"Loading SEG-Y file: {self.file_path}")
        return segfast.open(self.file_path, engine='segyio')  # for apple silicon

    def _load_headers(self) -> pd.DataFrame:
        logging.info("Loading headers from SEG-Y file")
        df = self.segfast_file.load_headers(['INLINE_3D', 'CROSSLINE_3D'])
        df.columns = ['TRACE_SEQUENCE_FILE', 'INLINE_3D', 'CROSSLINE_3D']
        logging.info(f"Loaded {len(df)} headers")
        return df

    def _get_dimensions(self) -> Tuple[int, int, int, int, int]:
        inline_min, inline_max = self.df['INLINE_3D'].min(), self.df['INLINE_3D'].max()
        crossline_min, crossline_max = self.df['CROSSLINE_3D'].min(), self.df['CROSSLINE_3D'].max()
        n_samples = self.segfast_file.n_samples
        return inline_min, inline_max, crossline_min, crossline_max, n_samples

    def _load_traces(self, traces_to_load: List[int]) -> np.ndarray:
        loaded_traces = []
        for trace in traces_to_load:
            try:
                loaded_trace = self.segfast_file.load_traces([trace], buffer=None)
                loaded_traces.append(loaded_trace[0])
            except OSError:
                logging.warning(f"Failed to load trace {trace}. Skipping.")
        logging.info(f"Loaded {len(loaded_traces)} out of {len(traces_to_load)} requested traces")
        return np.array(loaded_traces)


    def get_inline_slice(self, inline_index: int) -> np.ndarray:
        """
        Get a 2D slice along the inline direction.

        Args:
            inline_index (int): Index of the inline to retrieve.

        Returns:
            np.ndarray: 2D numpy array representing the inline slice.
        """
        traces_to_load = self.df[self.df['INLINE_3D'] == inline_index]['TRACE_SEQUENCE_FILE'].tolist()
        return self._load_traces(traces_to_load)

    def get_crossline_slice(self, crossline_index: int) -> np.ndarray:
        """
        Get a 2D slice along the crossline direction.

        Args:
            crossline_index (int): Index of the crossline to retrieve.

        Returns:
            np.ndarray: 2D numpy array representing the crossline slice.
        """
        traces_to_load = self.df[self.df['CROSSLINE_3D'] == crossline_index]['TRACE_SEQUENCE_FILE'].tolist()
        return self._load_traces(traces_to_load)

    def get_depth_slice(self, depth_index: int) -> np.ndarray:
        """
        Get a 2D slice at a specific depth.

        Args:
            depth_index (int): Index of the depth to retrieve.

        Returns:
            np.ndarray: 2D numpy array representing the depth slice.
        """
        depth_slice = self.segfast_file.load_depth_slices([depth_index], buffer=None)[0]
        return depth_slice.reshape(self.df['INLINE_3D'].nunique(), self.df['CROSSLINE_3D'].nunique())

    def get_slice(self, slice_type: str, slice_index: int) -> np.ndarray:
        logging.info(f"Requesting slice: {slice_type} = {slice_index}")
        if slice_type == 'INLINE_3D':
            traces_to_load = self.df[self.df['INLINE_3D'] == slice_index]['TRACE_SEQUENCE_FILE'].tolist()
        elif slice_type == 'CROSSLINE_3D':
            traces_to_load = self.df[self.df['CROSSLINE_3D'] == slice_index]['TRACE_SEQUENCE_FILE'].tolist()
        elif slice_type == 'DEPTH':
            return self.get_depth_slice(slice_index)
        else:
            raise ValueError("Invalid slice_type. Must be 'INLINE_3D', 'CROSSLINE_3D', or 'DEPTH'.")
        return self._load_traces(traces_to_load)
    def get_slice_range(self, slice_type: str) -> Tuple[int, int]:
        """
        Get the range of valid indices for a given slice type.

        Args:
            slice_type (str): Type of slice ('INLINE_3D', 'CROSSLINE_3D', or 'DEPTH').

        Returns:
            Tuple[int, int]: Minimum and maximum indices for the specified slice type.

        Raises:
            ValueError: If an invalid slice_type is provided.
        """
        if slice_type == 'INLINE_3D':
            return self.dimensions[0], self.dimensions[1]
        elif slice_type == 'CROSSLINE_3D':
            return self.dimensions[2], self.dimensions[3]
        elif slice_type == 'DEPTH':
            return 0, self.dimensions[4] - 1
        else:
            raise ValueError("Invalid slice_type. Must be 'INLINE_3D', 'CROSSLINE_3D', or 'DEPTH'.")

    def get_global_min_max(self, batch_size: int = 1000) -> Tuple[float, float]:
        """
        Get global minimum and maximum amplitude values using batched processing.

        Args:
            batch_size (int): Number of traces to process in each batch.

        Returns:
            Tuple[float, float]: Global minimum and maximum amplitude values.

        Raises:
            ValueError: If no valid data is found.
        """
        print("Calculating global min and max values. This may take a while...")
        global_min = float('inf')
        global_max = float('-inf')

        for i in range(0, len(self.df), batch_size):
            batch_traces = self.df['TRACE_SEQUENCE_FILE'].iloc[i:i + batch_size].tolist()
            batch_data = self._load_traces(batch_traces)
            if len(batch_data) > 0:
                batch_min = np.min(batch_data)
                batch_max = np.max(batch_data)
                global_min = min(global_min, batch_min)
                global_max = max(global_max, batch_max)
            print(f"Processed {i + len(batch_traces)} out of {len(self.df)} traces.")

        if global_min == float('inf') or global_max == float('-inf'):
            raise ValueError("Failed to calculate global min and max values. No valid data found.")

        print(f"Global min: {global_min}, Global max: {global_max}")
        return global_min, global_max

class SeismicVisualizer:
    def __init__(self, reader: SeismicSliceReader):
        self.reader = reader
        self.global_min, self.global_max = reader.get_global_min_max()
        self.cmap = 'seismic'
        self.setup_plot()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)

        initial_slice_type = 'INLINE_3D'
        initial_slice_value = (self.reader.dimensions[0] + self.reader.dimensions[1]) // 2
        seismic_slice = self.reader.get_slice(initial_slice_type, initial_slice_value)

        self.im = self.ax.imshow(seismic_slice.T, cmap=self.cmap, aspect='auto', vmin=self.global_min, vmax=self.global_max)
        self.cbar = plt.colorbar(self.im, ax=self.ax, label='Amplitude')
        self.title = self.ax.set_title(f"Seismic Slice ({initial_slice_type} = {initial_slice_value})")

        self.setup_widgets(initial_slice_type, initial_slice_value)

    def setup_widgets(self, initial_slice_type, initial_slice_value):
        slider_ax = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.slider = Slider(slider_ax, initial_slice_type, *self.reader.get_slice_range(initial_slice_type),
                             valinit=initial_slice_value, valstep=1)

        text_box_ax = plt.axes([0.1, 0.05, 0.2, 0.03])
        self.text_box = TextBox(text_box_ax, 'Slice Value', initial=str(initial_slice_value))

        rax = plt.axes([0.35, 0.05, 0.3, 0.05])
        self.radio = RadioButtons(rax, ('INLINE_3D', 'CROSSLINE_3D', 'DEPTH'))

        color_button_ax = plt.axes([0.66, 0.05, 0.1, 0.05])
        self.color_button = Button(color_button_ax, 'Switch Color')

        save_button_ax = plt.axes([0.77, 0.05, 0.1, 0.05])
        self.save_button = Button(save_button_ax, 'Save Slice')

        self.range_text = plt.text(0.1, 0.01, self.get_range_text(initial_slice_type), transform=self.fig.transFigure)

        self.slider.on_changed(self.update)
        self.text_box.on_submit(self.update_from_text)
        self.radio.on_clicked(self.update_slider_range)
        self.color_button.on_clicked(self.switch_color)
        self.save_button.on_clicked(self.save_current)

    def update(self, val):
        slice_type = self.radio.value_selected
        slice_value = int(self.slider.val)
        self.text_box.set_val(str(slice_value))
        seismic_slice = self.reader.get_slice(slice_type, slice_value)
        self.visualize_slice(seismic_slice, slice_type, slice_value)

    def update_from_text(self, text):
        try:
            val = int(text)
            self.slider.set_val(val)
        except ValueError:
            pass

    def update_slider_range(self, label):
        min_val, max_val = self.reader.get_slice_range(label)
        self.slider.valmin = min_val
        self.slider.valmax = max_val
        self.slider.ax.set_xlim(min_val, max_val)
        self.slider.valstep = max(1, (max_val - min_val) // 1000)  # Adjust step size
        self.slider.set_val((min_val + max_val) // 2)
        self.text_box.set_val(str((min_val + max_val) // 2))
        self.range_text.set_text(self.get_range_text(label))
        self.slider.label.set_text(label)
        self.update(self.slider.val)

    def switch_color(self, event):
        self.cmap = 'gray' if self.cmap == 'seismic' else 'seismic'
        self.update(self.slider.val)

    def save_current(self, event):
        slice_type = self.radio.value_selected
        slice_value = int(self.slider.val)
        seismic_slice = self.reader.get_slice(slice_type, slice_value)
        filename = f"./temp/seismic_slice_{slice_type}_{slice_value}.npy"
        np.save(filename, seismic_slice)
        print(f"Saved slice to {filename}")

    def visualize_slice(self, seismic_slice, slice_type, slice_value):
        self.im.set_data(seismic_slice.T if slice_type != 'DEPTH' else seismic_slice)
        self.im.set_cmap(self.cmap)
        self.title.set_text(f"Seismic Slice ({slice_type} = {slice_value})")
        if slice_type == 'DEPTH':
            self.ax.set_xlabel('INLINE_3D')
            self.ax.set_ylabel('CROSSLINE_3D')
        elif slice_type == 'INLINE_3D':
            self.ax.set_xlabel('CROSSLINE_3D')
            self.ax.set_ylabel('Time/Depth (samples)')
        else:  # CROSSLINE_3D
            self.ax.set_xlabel('INLINE_3D')
            self.ax.set_ylabel('Time/Depth (samples)')
        plt.draw()

    def get_range_text(self, slice_type):
        min_val, max_val = self.reader.get_slice_range(slice_type)
        return f"Range: {min_val} - {max_val}, Axis Size: {max_val - min_val + 1}"

    def show(self):
        plt.show()

def main():
    reader = SeismicSliceReader(SEGFAST_FILE_PATH)
    visualizer = SeismicVisualizer(reader)
    visualizer.show()

if __name__ == "__main__":
    main()