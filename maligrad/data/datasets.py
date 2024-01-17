from typing import Any, Literal
from pathlib import Path
import os
import requests

import numpy as np

from maligrad.data.engine import Dataset


class Moons(Dataset):

    def __init__(self, n_samples: int, seed: int | None = None) -> None:
        super().__init__()

        # generate 2 shifted circles
        fis = np.linspace(0, 2*np.pi, n_samples)
        xys = np.vstack([np.cos(fis), np.sin(fis)]).T
        upper = xys[xys[:, 1] > 0] + [0.5, 0]
        lower = xys[xys[:, 1] <= 0] + [-0.5, 0]
        
        # add noise to both coordinates
        if seed is not None:
            np.random.seed(seed)
        upper += np.random.normal(0, 0.5/3, upper.shape)
        lower += np.random.normal(0, 0.5/3, lower.shape)

        # save features and targets
        self.X = np.concatenate([upper, lower])
        self.y = np.array([1]*len(upper) + [0]*len(lower))

    def __getitem__(self, index: int) -> Any:
        return self.X[index], self.y[index]
    
    def __len__(self) -> int:
        return len(self.X)


class MNIST(Dataset):
    
    _url = "http://yann.lecun.com/exdb/mnist/{file}"

    _fname = "{subset}-{type}-ubyte.gz"

    _subsets = {
        "train": "train",
        "test": "t10k"
    }

    _types = {
        "images": "images-idx3",
        "labels": "labels-idx1"
    }

    _num_images = {
        "train": 60000,
        "test": 10000
    }

    _img_shape = (28, 28)

    def __init__(self, root: str, train: bool = True, download: bool = True) -> None:
        super().__init__()
        self.root = Path(root)
        self.subset = "train" if train else "test"
        
        if self.download:
            self.download(root)

        self.images, self.labels = self.load(self.subset)

    def __getitem__(self, index: int) -> Any:
        return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def load(self, subset: Literal["train", "test"]) -> tuple[np.ndarray, np.ndarray]:
        import gzip

        with gzip.open(self.root / self._fname.format(
            subset=self._subsets[subset],type=self._types["images"]
            )) as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).\
                reshape(self._num_images[subset], *self._img_shape)

        with gzip.open(self.root / self._fname.format(
            subset=self._subsets[subset], type=self._types["labels"]
            )) as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8).\
                reshape(self._num_images[subset])

        return images, labels

    @classmethod
    def download(cls, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        for subset in cls._subsets.values():
            for type_ in cls._types.values():
                fname = cls._fname.format(subset=subset, type=type_)
                response = requests.get(cls._url.format(file=fname))
                with open(Path(root) / fname, "wb") as f:
                    f.write(response.content)