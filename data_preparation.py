import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from SeismicSliceReader import SeismicSliceReader
from config import SEGFAST_FILE_PATH, TARGET_IMAGE_SIZE, BATCH_SIZE
from typing import Tuple, List
import random
from PIL import Image
import logging
import pickle


class SeismicDataset(Dataset):
    def __init__(self, file_path: str, slice_types: List[str] = ['INLINE_3D', 'CROSSLINE_3D'],
                 transform=None, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE):
        self.reader = SeismicSliceReader(file_path)
        self.slice_types = slice_types
        self.transform = transform
        self.target_size = target_size

        self.slices = []
        self.normalization_params = {}

        for slice_type in slice_types:
            min_val, max_val = self.reader.get_slice_range(slice_type)
            for i in range(min_val, max_val + 1):
                self.slices.append((slice_type, i))
                seismic_slice = self.reader.get_slice(slice_type, i)
                abs_max = max(abs(seismic_slice.min()), abs(seismic_slice.max()))
                self.normalization_params[(slice_type, i)] = abs_max

        # Сохраняем параметры нормализации
        self.save_normalization_params()

    def save_normalization_params(self):
        with open('normalization_params.pkl', 'wb') as f:
            pickle.dump(self.normalization_params, f)
        print("Normalization parameters saved to 'normalization_params.pkl'")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        slice_type, slice_index = self.slices[idx]
        seismic_slice = self.reader.get_slice(slice_type, slice_index)

        # Нормализация от -1 до 1
        abs_max = self.normalization_params[(slice_type, slice_index)]
        seismic_slice = seismic_slice / abs_max

        # Resize if target_size is specified
        if self.target_size:
            seismic_slice = self._resize_slice(seismic_slice, self.target_size)

        # Convert to tensor
        seismic_slice = torch.from_numpy(seismic_slice).float()

        if self.transform:
            seismic_slice = self.transform(seismic_slice)

        return seismic_slice

    @staticmethod
    def _resize_slice(slice: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        return np.array(Image.fromarray(slice).resize(target_size))


class PreloadedSeismicDataset(Dataset):
    def __init__(self, file_path: str, slice_types: List[str] = ['INLINE_3D', 'CROSSLINE_3D'],
                 transform=None, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE):
        self.reader = SeismicSliceReader(file_path)
        self.slice_types = slice_types
        self.transform = transform
        self.target_size = target_size

        self.slices = []
        self.data = []

        logging.info("Starting to preload data into memory...")
        for slice_type in slice_types:
            min_val, max_val = self.reader.get_slice_range(slice_type)
            for i in range(min_val, max_val + 1):
                self.slices.append((slice_type, i))
                seismic_slice = self.reader.get_slice(slice_type, i)
                seismic_slice = (seismic_slice - seismic_slice.min()) / (seismic_slice.max() - seismic_slice.min())
                if self.target_size:
                    seismic_slice = self._resize_slice(seismic_slice, self.target_size)
                self.data.append(torch.from_numpy(seismic_slice).float())
        logging.info(f"Finished preloading {len(self.data)} slices into memory.")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seismic_slice = self.data[idx]

        if self.transform:
            seismic_slice = self.transform(seismic_slice)

        return seismic_slice

    @staticmethod
    def _resize_slice(slice: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        return np.array(Image.fromarray(slice).resize(target_size))


def create_dataloader(dataset: Dataset, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def check_dataset(dataset: Dataset, batch_size: int = BATCH_SIZE):
    """
    Check and print information about the dataset and its batches.
    """
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)}")

    dataloader = create_dataloader(dataset, batch_size)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1} shape: {batch.shape}")
        if i == 0:  # Check only the first batch
            break


def main():
    # это косяк, файл нормализации не сохраняется вызовом PreloadedSeismicDataset
    dataset = SeismicDataset(SEGFAST_FILE_PATH)
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Normalization parameters saved for {len(dataset.normalization_params)} slices")


    # Используем PreloadedSeismicDataset вместо SeismicDataset
    dataset = PreloadedSeismicDataset(SEGFAST_FILE_PATH)
    check_dataset(dataset)

    # Визуализируем случайный образец
    random_index = random.randint(0, len(dataset) - 1)
    sample = dataset[random_index]
    plt.figure(figsize=(10, 8))
    plt.imshow(sample.numpy(), cmap='seismic', aspect='auto')
    plt.colorbar(label='Normalized Amplitude')
    plt.title(f"Sample {random_index}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    main()