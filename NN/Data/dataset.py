
from typing import Any, Optional, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from transformation import Transformation, ChainTransformation, ToNumpyArray
import pandas as pd
import numpy as np
from PIL import Image

class Dataset(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define __getitem__() and __len__().

    Args:
        root: Storing directory of dataset
        transforms: Transformations which should be applied to the data
    """

    def __init__(self, root: Path, transforms: Optional[Transformation]=ToNumpyArray):

        self.root = root
        self.transforms = transforms

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Returns one sample of the dataset.

        Args:
            index: Index of the sample, which should be returned.

        Returns:
            Any: Sample
        """
        pass


class MNISTDataset(Dataset):

    def __init__(self, root: Path, image_dir_name: str, labels_file_name: str, transforms: Optional[Transformation]):
        """
        MNIST-style dataset loading images and labels from disk.

        Args:
            root: Root directory of the dataset.
            image_dir_name: Folder name containing images.
            labels_file_name: CSV or file containing labels.
            transforms: Optional transforms applied to images.
        """

        super().__init__(root=root, transforms=transforms)

        self.labels = self.load_labels(labels_dir=Path(root/labels_file_name))

        self.images = root / image_dir_name

    @staticmethod
    def load_labels(labels_dir: Path) -> List:
        """
        Load the labels of a csv file and return a list of labels.

        Args:
             labels_dir: Path to the labels file

        Returns:
            List: Labels list
        """

        labels_df = pd.read_csv(labels_dir)
        labels = labels_df['C2'].tolist()

        return labels

    @staticmethod
    def load_image_as_numpy_array(image_path: Path) -> np.ndarray:
        """
        Loads an PIL image and convert it into a numpy array.

        Args:
             image_path: Path to the PIL image

         Returns:
             np.ndarray: Converted PIL image
        """

        return np.asarray(Image.open(image_path))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Any:

        image_path = self.images / f"{index}.png"

        label = self.labels[index] # class of image

        image_array = self.transforms.transform(image_path)

        assert image_array.dtype == np.uint8, f"Image must be a numpy array! Convert first with {ToNumpyArray}"

        return image_array, label