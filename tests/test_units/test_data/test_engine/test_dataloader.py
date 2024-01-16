from typing import Any

import numpy as np

from maligrad.data.engine import DataLoader, Dataset


class Dummy(Dataset):

    def __init__(self, len: int) -> None:
        super().__init__()
        self._len = len
    
    def __getitem__(self, index: int) -> Any:
        return index, index+1
    
    def __len__(self) -> int:
        return self._len


def test_simple():
    batch_size = 8
    loader = DataLoader(Dummy(32), 8, shuffle=False)
    for i, batch in enumerate(loader):
        x, y = batch
        assert np.all(i*batch_size + np.arange(batch_size) == x)
        assert np.all(i*batch_size + 1 + np.arange(batch_size) == y)