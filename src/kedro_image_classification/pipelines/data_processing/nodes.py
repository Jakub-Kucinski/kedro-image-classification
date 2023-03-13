import torch


def load_dataset(loaders_config, cifar_dataset):
    train_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=loaders_config["train_loader"]["batch_size"],
        shuffle=loaders_config["train_loader"]["shuffle"],
        num_workers=loaders_config["train_loader"]["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=loaders_config["test_loader"]["batch_size"],
        shuffle=loaders_config["test_loader"]["shuffle"],
        num_workers=loaders_config["test_loader"]["num_workers"],
    )
    return train_loader, test_loader
