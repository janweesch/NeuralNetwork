import numpy
from numpy import ndarray, dtype
from dataset import Dataset
import numpy as np
from typing import Tuple, Any, Generator
from numpy._core.multiarray import _ScalarT

class Dataloader:
    """
    Loads data with targets from the dataset.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool=False, drop_last: bool=False):
        """
        Args:
             dataset: Dataset where the data should be loaded from
             batch_size: Number of samples to load before updating model parameters
             shuffle: Shuffle indexes of dataset
             drop_last: Drop last samples if not completely loaded from dataset
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle


    def __iter__(self) -> Generator[
        tuple[ndarray[tuple[Any, ...], dtype[_ScalarT]], ndarray[tuple[Any, ...], dtype[_ScalarT]]], Any, None]:
        """
         Iterates over dataset and retrieves images and labels as batch.

         Returns:
             Generator[...]: Batch with images and labels
        """

        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_iterator = iter(range(len(self.dataset)))

        images_batch = []
        labels_batch = []

        # retrieve images and labels from dataset
        for index in index_iterator:
            image, label = self.dataset[index]
            images_batch.append(image)
            labels_batch.append(label)

            # return batch
            if len(images_batch) == self.batch_size:
                # convert to numpy array for preprocessing
                images_batch_array = np.array(images_batch)
                labels_batch_array = np.array(labels_batch)
                yield images_batch_array, labels_batch_array
                images_batch = []
                labels_batch = []

        # return remaining samples
        if len(images_batch) > 0 and not self.drop_last:
            images_batch_array = np.array(images_batch)
            labels_batch_array = np.array(labels_batch)
            yield images_batch_array, labels_batch_array

    def len(self) -> int:
        """
        Returns length of dataloader.

        Returns:
            int: Length of dataloader
        """

        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = int(np.ceil(len(self.dataset) / self.batch_size))

        return length


