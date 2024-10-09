import unittest
import numpy as np
from src.SeismicSliceReader import SeismicSliceReader
from src.config import SEGFAST_FILE_PATH


class TestSeismicSliceReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = SeismicSliceReader(SEGFAST_FILE_PATH)

    def test_initialization(self):
        self.assertIsNotNone(self.reader)
        self.assertIsNotNone(self.reader.segfast_file)
        self.assertIsNotNone(self.reader.df)
        self.assertIsNotNone(self.reader.dimensions)

    def test_dimensions(self):
        inline_min, inline_max, crossline_min, crossline_max, n_samples = self.reader.dimensions
        self.assertIsInstance(inline_min, int)
        self.assertIsInstance(inline_max, int)
        self.assertIsInstance(crossline_min, int)
        self.assertIsInstance(crossline_max, int)
        self.assertIsInstance(n_samples, int)
        self.assertLess(inline_min, inline_max)
        self.assertLess(crossline_min, crossline_max)
        self.assertGreater(n_samples, 0)

    def test_get_inline_slice(self):
        inline_min, inline_max, _, _, n_samples = self.reader.dimensions
        inline_index = (inline_min + inline_max) // 2
        inline_slice = self.reader.get_inline_slice(inline_index)
        self.assertIsInstance(inline_slice, np.ndarray)
        self.assertEqual(inline_slice.ndim, 2)

    def test_get_crossline_slice(self):
        _, _, crossline_min, crossline_max, n_samples = self.reader.dimensions
        crossline_index = (crossline_min + crossline_max) // 2
        crossline_slice = self.reader.get_crossline_slice(crossline_index)
        self.assertIsInstance(crossline_slice, np.ndarray)
        self.assertEqual(crossline_slice.ndim, 2)

    def test_get_depth_slice(self):
        _, _, _, _, n_samples = self.reader.dimensions
        depth_index = n_samples // 2
        depth_slice = self.reader.get_depth_slice(depth_index)
        self.assertIsInstance(depth_slice, np.ndarray)
        self.assertEqual(depth_slice.ndim, 2)

    def test_get_slice(self):
        inline_min, inline_max, crossline_min, crossline_max, n_samples = self.reader.dimensions

        inline_slice = self.reader.get_slice('INLINE_3D', (inline_min + inline_max) // 2)
        self.assertIsInstance(inline_slice, np.ndarray)
        self.assertEqual(inline_slice.ndim, 2)

        crossline_slice = self.reader.get_slice('CROSSLINE_3D', (crossline_min + crossline_max) // 2)
        self.assertIsInstance(crossline_slice, np.ndarray)
        self.assertEqual(crossline_slice.ndim, 2)

        depth_slice = self.reader.get_slice('DEPTH', n_samples // 2)
        self.assertIsInstance(depth_slice, np.ndarray)
        self.assertEqual(depth_slice.ndim, 2)

    def test_get_slice_range(self):
        for slice_type in ['INLINE_3D', 'CROSSLINE_3D', 'DEPTH']:
            min_val, max_val = self.reader.get_slice_range(slice_type)
            self.assertIsInstance(min_val, int)
            self.assertIsInstance(max_val, int)
            self.assertLess(min_val, max_val)

    def test_get_global_min_max(self):
        global_min, global_max = self.reader.get_global_min_max()
        self.assertIsInstance(global_min, float)
        self.assertIsInstance(global_max, float)
        self.assertLess(global_min, global_max)

    def test_invalid_slice_type(self):
        with self.assertRaises(ValueError):
            self.reader.get_slice('INVALID_TYPE', 0)

    def test_out_of_range_slice_index(self):
        inline_min, inline_max, _, _, _ = self.reader.dimensions
        with self.assertRaises(Exception):  # The specific exception might vary
            self.reader.get_slice('INLINE_3D', inline_max + 1)


if __name__ == '__main__':
    unittest.main()