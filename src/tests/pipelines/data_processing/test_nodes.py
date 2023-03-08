import torch

from kedro_image_classification.pipelines.data_processing.nodes import load_dataset


def test_dataset_loading():
    """Test checking load_dataset method."""
    cfg = {
        "train_loader": {"batch_size": 4, "shuffle": True, "num_workers": 4},
        "test_loader": {"batch_size": 4, "shuffle": False, "num_workers": 4},
    }

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    loaded_dataset = load_dataset(cfg, tensor)

    assert len(loaded_dataset) == 2
    assert type(loaded_dataset[0]) == torch.utils.data.DataLoader
    assert type(loaded_dataset[1]) == torch.utils.data.DataLoader
