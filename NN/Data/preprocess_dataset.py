from PIL import Image
import pandas as pd
import numpy as np
import struct
from array import array
from typing import List, Tuple, Optional
from pathlib import Path
import kagglehub


def download_kaggle_dataset(kaggle_path: str="hojjatk/mnist-dataset", save_dir_dataset: str="dataset") -> Path:
    """
    Download a dataset from kaggle. The dataset direction of the MNIST dataset on kaggle is predefined.

    Args:
        kaggle_path: Path to the dataset on kaggle.
        save_dir_dataset: Path to the local saved dataset.

    Returns:
        Path: Directory containing the downloaded dataset.

    """
    # Download latest version
    path = kagglehub.dataset_download(handle=kaggle_path, output_dir=save_dir_dataset)

    path = Path(path)

    return path

def convert_byte_dataset(images_byte_filepath: Path, labels_byte_filepath: Path, save_dir_dataset: Path, images_dir_name: str = "images", labels_file_name: str = "labels") -> Path:
    """
    Converts MNIST-style byte files into a usable dataset format.

    This function reads image and label IDX byte files, converts the images
    into PIL images, and stores them in a structured directory along with
    a CSV file containing the corresponding labels.

    Args:
        images_byte_filepath: Path to the IDX image byte file.
        labels_byte_filepath: Path to the IDX label byte file.
        save_dir_dataset: Root directory where the processed dataset will be stored.
        images_dir_name: Subdirectory name for saving images (default: "images").
        labels_file_name: Filename (without extension) for the labels CSV (default: "labels").

    Returns:
        Path: Path to the created dataset directory.
    """

    save_dir_dataset.mkdir(parents=True, exist_ok=True)
    labels_file = save_dir_dataset/f"{labels_file_name}.csv"

    images_dir = save_dir_dataset/images_dir_name
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_byte_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())
        labels = np.array(labels)
        df = pd.DataFrame(labels)
        df.to_csv(labels_file, index=True)

    with open(images_byte_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())

    # store images as PIL image
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        img = Image.fromarray(img)  # convert to PIL Image
        img.save(images_dir / f"{i}.png")

    return save_dir_dataset

if __name__ == "__main__":
    download_kaggle_dataset(save_dir_dataset="byte_dataset")
    convert_byte_dataset(images_byte_filepath=Path("byte_dataset/train-images.idx3-ubyte"), labels_byte_filepath=Path("byte_dataset/train-labels.idx1-ubyte"), save_dir_dataset=Path("dataset/train"))
    convert_byte_dataset(images_byte_filepath=Path("byte_dataset/t10k-images.idx3-ubyte"), labels_byte_filepath=Path("byte_dataset/t10k-labels.idx1-ubyte"), save_dir_dataset=Path("dataset/test"))
