import pytest
import torch
from kedro.framework.context import KedroContext
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import kedro_image_classification.pytorch.models as models
from kedro_image_classification.pipelines.model_training.nodes import (
    create_model,
    create_task,
    create_trainer,
    train,
)
from kedro_image_classification.pytorch.tasks import ClassificationTask


def test_model_creation(project_context: KedroContext):
    """Test checking create_model method from model_training pipeline."""

    project_catalog = project_context.catalog
    model_architectures = project_catalog.load("params:model_architectures")
    model_selection = project_catalog.load("params:model_selection")

    model = create_model(model_architectures, model_selection)

    assert type(model) == models.CustomConvModel

    model_selection["model_name"] = "NonExistingModel"

    with pytest.raises(KeyError):
        create_model(model_architectures, model_selection)

    model_architectures["EmptyModel"] = {"model_type": "None"}
    model_selection["model_name"] = "EmptyModel"

    with pytest.raises(AttributeError):
        create_model(model_architectures, model_selection)


def test_trainer_creation(project_context: KedroContext):
    """Test checking create_trainer method from model_training pipeline."""

    project_catalog = project_context.catalog
    trainer_params = project_catalog.load("params:trainer_params")

    trainer = create_trainer(trainer_params)

    assert type(trainer) == Trainer
    assert trainer.num_devices == trainer_params["devices"]


def test_task_creation(project_context: KedroContext):
    """Test checking create_task method from model_training pipeline."""
    project_catalog = project_context.catalog
    model_architectures = project_catalog.load("params:model_architectures")
    model_selection = project_catalog.load("params:model_selection")

    model = create_model(model_architectures, model_selection)

    optimizer_params = project_catalog.load("params:optimizer_params")

    task = create_task(model, optimizer_params)

    assert type(task) == ClassificationTask


def test_train(project_context: KedroContext):
    """Test checking train method from model_training pipeline."""
    train_data = torch.zeros(10, 3, 32, 32)
    labels = torch.tensor([1, 2, 3, 4, 4, 5, 2, 1, 6, 7])

    dataset = torch.utils.data.TensorDataset(train_data, labels)

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    trainer = Trainer(max_epochs=2)

    catalog = project_context.catalog

    model = create_model(
        model_architectures=catalog.load("params:model_architectures"),
        model_selection=catalog.load("params:model_selection"),
    )

    task = create_task(
        model=model, optimizer_params=catalog.load("params:optimizer_params")
    )

    trainer = train(trainer, task, train_loader, train_loader)
    assert type(trainer) == Trainer
