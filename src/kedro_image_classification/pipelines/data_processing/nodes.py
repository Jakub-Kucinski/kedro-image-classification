from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10


def load_dataset(
    loaders_config: dict, cifar_dataset: Tuple[CIFAR10, CIFAR10]
) -> Tuple[DataLoader, DataLoader]:
    """Function creating data loaders for train and test set
    from given loaders configuration and dataset.

    Args:
        loaders_config (dict): Dictionary with configuration of train_set
        and test_set loaders.
        cifar_dataset (Tuple[CIFAR10, CIFAR10]): Downloaded torchvision cifar dataset.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple consisting of train_set
        and test_set data loaders.
    """
    train_data, test_data = cifar_dataset
    train_loader = DataLoader(
        train_data,
        batch_size=loaders_config["train_loader"]["batch_size"],
        shuffle=loaders_config["train_loader"]["shuffle"],
        num_workers=loaders_config["train_loader"]["num_workers"],
    )
    test_loader = DataLoader(
        test_data,
        batch_size=loaders_config["test_loader"]["batch_size"],
        shuffle=loaders_config["test_loader"]["shuffle"],
        num_workers=loaders_config["test_loader"]["num_workers"],
    )
    return train_loader, test_loader
