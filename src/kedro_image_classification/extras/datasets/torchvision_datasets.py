import os
from pathlib import PurePosixPath

import fsspec
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path
from torchvision import datasets, transforms


class TorchvisionCIFAR10(AbstractDataSet):
    def __init__(self, path, save_args):
        protocol, path = get_protocol_and_path(path)
        self._protocol = protocol
        self._path = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self.supported_clouds = ["aws"]
        self.download_source = save_args["source"]

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

    def _save(self, data):
        def torchvision_download(save_path):
            _ = datasets.CIFAR10(save_path, download=True)

        def aws_download(save_path):
            os.system("dvc pull")
            os.system(
                "tar -xf " + save_path + "/cifar-10-python.tar.gz -C " + save_path
            )
            os.system("rm " + save_path + "/cifar-10-python.tar.gz")

        def cloud_download(cloud, save_path):
            if cloud == "aws":
                aws_download(save_path)

        save_path = get_filepath_str(self._path.parent, self._protocol)
        files = os.listdir(save_path)
        if "cifar-10-batches-py" in files:
            print("Files already downloaded")
            return
        if self.download_source == "torchvision":
            torchvision_download(save_path)
        else:
            if self.download_source in self.supported_clouds:
                cloud_download(self.download_source, save_path)
            else:
                raise Exception(
                    f"Selected download from cloud, but provided cloud name is not "
                    f"supported.\nList of supported clouds: {self.supported_clouds}"
                )

    def _describe(self):
        return dict(filepath=self._path, protocol=self._protocol)
