from typing import Any
from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    

class DataLoader:
    # TODO: add shuffling, determinism, change so all data is batched each epoch

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int | None = None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._indices = np.arange(len(dataset))

    def __iter__(self):
        for i in range(0, len(self._indices), self.batch_size):
            yield self._gather_batch(i, i+self.batch_size)
    
    def _gather_batch(self, start_idx, stop_idx):
        batch_iter = (
            self.dataset[self._indices[idx]]
            for idx in range(start_idx, stop_idx)
        )
        return [np.stack(e, axis=0) for e in zip(*batch_iter)]