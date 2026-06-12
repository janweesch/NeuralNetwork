from PIL import Image
import pandas as pd
import numpy as np
import struct
from array import array
from typing import List, Tuple, Optional
from pathlib import Path
import requests
from tqdm.auto import tqdm
import tarfile


def download_MNIST_dataset(images_url: str, labels_url: str, save_dir_dataset: Path):
    """
    Downloads the MNIST dataset from the remote URLs if not already present locally and unzipped the images.
    It first checks the existence of the dataset in the specific directory. If the files are not found, it proceeds to download them from the pre-defined
    URLs into the storing direction.

    Args:
        images_url: URL for the image archive
        labels_url: URL for the label archive
        save_dir_dataset: Directory of downloaded dataset
    """

    # Define paths for zipped dataset
    save_dir_tgz_images = save_dir_dataset / "tgz" / "images"
    save_dir_tgz_labels = save_dir_dataset / "tgz" / "labels"

    # Define pats for unzipped dataset
    save_dir_images = save_dir_dataset / "unzipped" / "images"
    save_dir_labels = save_dir_dataset / "unzipped" / "labels"

    # check if dataset is already downloaded
    if save_dir_tgz_images.exists() and save_dir_tgz_labels.exists():
        print(f"Dataset already downloaded. Found zipped version at '{save_dir_tgz_images}' and '{save_dir_tgz_labels}'.")
        return

    # Start Download
    print("Dataset not found locally. Downloading...")

    save_dir_tgz_images.mkdir() # create directory if it does not exist

    print("Downloading images...")

    response = requests.get(images_url, stream=True) # stream the image files from the website

    total_size = int(response.headers.get("content-length", 0)) # Get the total size of the file from the response headers for the progress bar

    chunk_size = 1024
    with open(save_dir_tgz_images, "wb") as file: # open as binary file
        for data in tqdm(response.iter_content(chunk_size=chunk_size), total=total_size//chunk_size):
            file.write(data)

    print("Extracting images...")

    with tarfile.open(save_dir_tgz_images, "r:gz") as tar: # extract all images
        tar.extractall(save_dir_images)

    print ("Images extracted...")

    print("Downloading labels...")

    response = requests.get(labels_url, stream=True)  # stream the image files from the website

    total_size = int(response.headers.get("content-length",
                                          0))  # Get the total size of the file from the response headers for the progress bar
    chunk_size = 1024

    with open(save_dir_tgz_labels, "wb") as file: # open as binary file
        for data in tqdm(response.iter_content(chunk_size=chunk_size), total=total_size//chunk_size):
            file.write(data)

    with tarfile.open(save_dir_tgz_labels, "r:gz") as tar: # extract all images
        tar.extractall(save_dir_labels)

    print("Labels extracted...")

    print(f"Dataset downloaded to '{save_dir_tgz_images}' and '{save_dir_tgz_labels}'.")

def load_byte_dataset(images_filepath, labels_filepath) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load a pre downloaded byte dataset as arrays.
    Code copied from Kaggle (https://www.kaggle.com/code/hojjatk/read-mnist-dataset)

    Args:
        images_filepath: Path to the byte image folder
        labels_filepath: Path to the byte label folder

    Returns:
        images: List of images as numpy arrays
        labels: labels as numpy arrays
    """

    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())
        labels = np.array(labels)

    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img

    return images, labels

def save_dataset(images: List[np.ndarray], labels: np.ndarray, save_dir_images: Path="images", save_dir_labels: Path="labels") -> Tuple[Path, Path]:
    """
    Save Images as PIL Image and Labels as comma seperated values.

    Args:
        images: Images as list of numpy arrays
        labels: labels as numpy arrays
        save_dir_images: Path direction for saving images
        save_dir_labels: Path direction for saving labels

    Returns:
        save_dir_images: Path direction of saved images
        save_dir_labels: Path direction of saved labels
    """

    # Save Images
    save_dir_images.mkdir() # create path
    for i, img_array in enumerate(images):
        img = Image.fromarray(img_array) # convert to PIL Image
        img.save(save_dir_images / f"{i}.png")

    # Save labels
    save_dir_labels.mkdir() # create path
    labels = pd.DataFrame(labels)
    labels.to_csv(save_dir_labels)

    return save_dir_images, save_dir_labels