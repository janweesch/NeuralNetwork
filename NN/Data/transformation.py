import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Any, List
from abc import ABC, abstractmethod

def calculate_mean_std(image: np.ndarray, axis:Tuple[int]=(1,2)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and standard deviation of an image.

    Args:
         image: Pixel values as numpy array
         axis: Over which axis to calculate

     Returns:
         Tuple[float, float]: Mean and standard deviation of image
    """

    mean = np.mean(image, axis=axis, keepdims=True) # dimensions of output array should match the input
    std = np.std(image, axis=axis, keepdims=True)

    return np.array(mean), np.array(std)

class Transformation(ABC):
    """
    Base Class for transformations, which are applied as image preprocess step
    """

    @abstractmethod
    def transform(self, data: Any) -> np.ndarray:
        """
        Applies a transformation on the image and returns the transformed image.

        Args:
            data: Image

        Returns:
            np.ndarray: Image after transformation
        """
        pass

    def __call__(self, data: Any):
        return self.transform(data)

class ChainTransformation(Transformation):
    """
    Chain different transformations.
    """

    def __init__(self, transformations: List[Transformation]):
        """
        Args:
             transformations: Transformations to apply
        """""
        self.transformations = transformations

    def transform(self, data):
        for transformation in self.transformations:
            data = transformation.transform(data=data)

        return data

class ToNumpyArray(Transformation):
    """
    Converts a PIL image into a Numpy array.
    """

    def transform(self, data: Path) -> np.ndarray:
        """
        Loads an PIL image and convert it into a numpy array.
        Changes the axis of the image into the order [C, H, W].

        Args:
             data: Path to the PIL image

         Returns:
             np.ndarray: Converted PIL image
        """

        image = np.asarray(Image.open(data))
        image = np.transpose(image, (-1, 0, 1)) # Height, Width, Channels -> Channels, Height, Width

        return  image # Channels, Height, Width

class Normalize(Transformation):
    """
    Normalizes the pixel values of an image to a defined output range.
    """

    def __init__(self, in_range=(0, 255), out_range=(0, 1)):

        self.in_range_min = min(in_range)
        self.in_range_max = min(in_range)

        self.out_range_min = min(out_range)
        self.out_range_max = max(out_range)

    def transform(self, data: np.ndarray):

        assert data.dtype == np.uint8, "Image is not a numpy array!"

        # transforms the pixel values to the given output range
        old_range = self.in_range_max - self.in_range_min
        if old_range == 0:
            return np.zeros(data.shape)
        else:
            new_range = self.out_range_max - self.out_range_min
            normalized_image = (((data.copy() - self.in_range_min) * new_range) / old_range) +  self.out_range_min

            return normalized_image

class Standardization(Transformation):
    """
    Shifts the image to mean zero and sets the standard deviation of pixel values to 1
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        """
        Args:
             mean: Mean of each channel
             std: Standard deviation of each channel
        """

        self.mean = mean
        self.std = std

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Standardize an image around zero with std = 1.

        Args:
            data: Numpy array of pixel values

        Returns:
            np.ndarray: Standardized image
        """

        # reshape the array for broadcasting for each channel

        assert data.shape[0] == self.mean.shape[0] and data.shape[0] == self.std.shape[0], "The shape of Image channels is not the same as the number of calculated means and standard deviations for the image."

        std_image = (data - self.mean) / self.std

        return std_image
