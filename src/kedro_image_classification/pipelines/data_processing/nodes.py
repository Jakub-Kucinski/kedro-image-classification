from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose


def create_transforms(transforms_config: dict) -> Tuple[list, list]:
    """Function creating transforms from given configuration.

    Args:
        transforms_config (dict): Dictionary with config of train and test transforms.

    Returns:
        train_transforms: List of created train transforms.
        test_transforms: List of created test transforms.
    """

    def config_to_transforms(transforms_config: list) -> list:
        transforms_list = []
        for transformation in transforms_config:
            if isinstance(transformation, str):
                print(transformation)
                transformation_class = getattr(transforms, transformation)
                transforms_list.append(transformation_class())
            elif isinstance(transformation, dict):
                transformation_name = next(iter(transformation))
                transformation_class = getattr(transforms, transformation_name)
                params = transformation[transformation_name]
                if isinstance(params, dict):
                    transforms_list.append(transformation_class(**params))
                elif isinstance(params, list):
                    transforms_list.append(transformation_class(*params))
        return transforms_list

    train_transforms = config_to_transforms(transforms_config["train_transforms"])
    test_transforms = config_to_transforms(transforms_config["test_transforms"])
    return train_transforms, test_transforms


def load_dataset(
    loaders_config: dict,
    train_transforms: list,
    test_transforms: list,
    cifar_dataset: Tuple[CIFAR10, CIFAR10],
) -> Tuple[DataLoader, DataLoader]:
    """Function creating data loaders for train and test set
    from given loaders configuration and dataset.

    Args:
        loaders_config (dict): Dictionary with configuration of train_set
        and test_set loaders.
        train_transforms (list): List of train transforms.
        test_transforms (list): List of test transforms.
        cifar_dataset (Tuple[CIFAR10, CIFAR10]): Downloaded torchvision cifar dataset.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple consisting of train_set
        and test_set data loaders.
    """
    train_data, test_data = cifar_dataset
    train_data.transforms = Compose(train_transforms)
    test_data.transforms = Compose(test_transforms)
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
