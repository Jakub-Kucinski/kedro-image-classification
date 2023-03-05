from pathlib import PurePosixPath

import fsspec
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path
from torchvision import datasets, transforms


class TorchvisionCIFAR10(AbstractDataSet):
    def __init__(self, path):
        protocol, path = get_protocol_and_path(path)
        self._protocol = protocol
        self._path = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self):
        load_path = get_filepath_str(self._path.parent, self._protocol)
        cifar_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train_data = datasets.CIFAR10(
            load_path, train=True, download=False, transform=cifar_transforms
        )
        test_data = datasets.CIFAR10(
            load_path, train=False, download=False, transform=cifar_transforms
        )
        return train_data, test_data

    def _save(self):
        save_path = get_filepath_str(self._path, self._protocol)
        _ = datasets.CIFAR10(save_path, download=True)

    def _describe(self):
        return dict(filepath=self._path, protocol=self._protocol)
