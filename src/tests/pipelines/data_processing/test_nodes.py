import pytest
import torch
from torchvision import transforms

from kedro_image_classification.pipelines.data_processing.nodes import load_dataset


def test_dataset_loading():
    """Test checking load_dataset method."""
    cfg = {
        "train_loader": {"batch_size": 4, "shuffle": True, "num_workers": 4},
        "test_loader": {"batch_size": 4, "shuffle": False, "num_workers": 4},
    }

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    img_transforms = [
        transforms.ToTensor(),
    ]
    train_loader, test_loader = load_dataset(
        cfg, img_transforms, img_transforms, tensor
    )

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(),
        {"train_loader": {"batch_size": 4, "shuffle": True, "num_workers": 4}},
        {"test_loader": {"batch_size": 4, "shuffle": False, "num_workers": 4}},
        {
            "train_loader": {"shuffle": True, "num_workers": 4},
            "test_loader": {"batch_size": 4, "shuffle": False, "num_workers": 4},
        },
    ],
)
def test_load_dataset_raises_keyerror(cfg):
    """
    Test checking load_dataset method raises KeyError when test_loader is not in cfg.
    """
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    img_transforms = [
        transforms.ToTensor(),
    ]
    with pytest.raises(KeyError):
        load_dataset(cfg, img_transforms, img_transforms, tensor)


@pytest.mark.parametrize(
    "cfg",
    [
        "train_loader",
        1,
        1.0,
        True,
        None,
        ["train_loader"],
        (1, 2, 3),
        {1, 2, 3},
    ],
)
def test_load_dataset_raises_typeerror(cfg):
    """Test checking load_dataset method raises TypeError when cfg is not a dict."""
    cfg = "fdsfsd"
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    img_transforms = [
        transforms.ToTensor(),
    ]
    with pytest.raises(TypeError):
        load_dataset(cfg, img_transforms, img_transforms, tensor)
