from typing import Dict, List, Tuple, Union

import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
import random

from torchvision.transforms import functional as F
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.transforms.rotation import random_rotation_transform

class GaussianBlur:
    def __init__(self, prob=0.5, sigmas=(0.1, 2.0), kernel_size=5):
        self.prob = prob
        self.sigma_min = sigmas[0]
        self.sigma_max = sigmas[1]
        self.kernel_size = kernel_size

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        return F.gaussian_blur(
            img,
            kernel_size=self.kernel_size,
            sigma=(self.sigma_min, self.sigma_max),
        )


"""class MAETransform:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
        normalize: Dict[str, List[float]] = IMAGENET_NORMALIZE,
    ):
        transforms = [
            T.RandomCrop(
                input_size, 
                # scale=(min_scale, 1.0), interpolation=3
            ),  # 3 is bicubic
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
        if normalize:
            transforms.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))

        self.transform = T.Compose(transforms)"""

class MAETransform:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        normalize: Dict[str, List[float]] = IMAGENET_NORMALIZE,
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees=None,          # None => 90-degree rotations in lightly helper
        blur_prob: float = 1.0,
        blur_kernel_size = 5,
        blur_sigmas=(0.1, 2.0),
    ):
        transforms = [
            T.RandomCrop(input_size),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            GaussianBlur(
                prob=blur_prob,
                sigmas=blur_sigmas,
                kernel_size=blur_kernel_size,
            ),
        ]

        if normalize:
            transforms.append(
                T.Normalize(mean=normalize["mean"], std=normalize["std"])
            )

        self.transform = T.Compose(transforms)



    def __call__(self, image: Union[Tensor, Image]) -> List[Tensor]:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        return [self.transform(image)]