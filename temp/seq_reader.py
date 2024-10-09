import numpy as np
import segfast
import matplotlib.pyplot as plt

class SeismicReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.segy_file = segfast.open(file_path, engine='memmap')
        self.segy_file.load_headers(['INLINE_3D', 'CROSSLINE_3D'])
        self.segy_file.convert(format=5)  # Convert to IEEE float32 for better performance
        
    def get_slice(self, index, axis):
        """
        Получает срез сейсмических данных.
        
        :param index: индекс среза
        :param axis: ось среза ('inline', 'crossline', или 'time')
        :return: numpy.ndarray с данными среза
        """
        if axis == 'inline':
            return self.segy_file.load_inline(index)
        elif axis == 'crossline':
            return self.segy_file.load_crossline(index)
        elif axis == 'time':
            return self.segy_file.load_depth_slice(index)
        else:
            raise ValueError("Axis must be 'inline', 'crossline', or 'time'")
    
    def visualize_slice(self, slice_data):
        """
        Визуализирует срез сейсмических данных.
        
        :param slice_data: numpy.ndarray с данными среза
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(slice_data, cmap='seismic', aspect='auto')
        plt.colorbar(label='Amplitude')
        plt.title('Seismic Slice')
        plt.xlabel('Trace')
        plt.ylabel('Time/Depth')
        plt.show()

    def get_dimensions(self):
        """
        Возвращает размерности сейсмического куба.
        
        :return: кортеж (n_inlines, n_crosslines, n_samples)
        """
        n_inlines = len(np.unique(self.segy_file.headers['INLINE_3D']))
        n_crosslines = len(np.unique(self.segy_file.headers['CROSSLINE_3D']))
        n_samples = self.segy_file.samples_count
        return (n_inlines, n_crosslines, n_samples)

# Пример использования
reader = SeismicReader('/Users/admin/projects/data/seismic.sgy')

# Получение размерностей куба
dimensions = reader.get_dimensions()
print(f"Cube dimensions: {dimensions}")

# Получение inline среза
inline_index = min(100, dimensions[0] - 1)  # Убедимся, что индекс не выходит за границы
inline_slice = reader.get_slice(inline_index, 'inline')

# Визуализация среза
reader.visualize_slice(inline_slice)