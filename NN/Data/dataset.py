
from typing import Any, Optional, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from transformation import Transformation

class Dataset(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define __getitem__() and __len__().

    Args:
        root: Storing directory of dataset
        transforms: Transformations which should be applied to the data
    """

    def __init__(self, root: Path, annotations_file: Optional[Path], transforms: Optional[Transformation]=None):

        self.root = root
        self.annotations_file = annotations_file
        self.transforms = transforms

    @abstractmethod
    def __len__(self):
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

    def __init__(self, image_path: Path, annotations_path: Path, transforms: Optional[Transformation]=None):
        super().__init__(root=image_path, annotations_file=annotations_path, transforms=transforms)

        self.images = image_path
        self.labels = annotations_path


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index:int):

        if self.transforms:
            pass









