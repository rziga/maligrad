from typing import Any
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