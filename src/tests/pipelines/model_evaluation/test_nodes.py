import pytest
import torch
from kedro.framework.context import KedroContext
from torch.utils.data import DataLoader

from kedro_image_classification.pipelines.model_evaluation.nodes import (
    calc_metrics,
    get_confusion_matrix,
    get_metric_value,
    load_testset,
)

# from kedro_image_classification.pipelines.model_training.nodes import (
#     create_model,
#     create_task,
# )


def test_load_testset(project_context: KedroContext):
    """Test checking load_testset method from model_evaluation pipeline"""

    project_catalog = project_context.catalog
    loaders_config = project_catalog.load("params:loaders")

    dataset = project_catalog.load("CIFAR10")

    loader = load_testset(loaders_config, dataset)

    assert type(loader) == DataLoader


# def test_make_predictions(project_context: KedroContext):
#     """Test checking make_prediction method from model_evaluation_pipeline
#     """

#     project_catalog = project_context.catalog

#     task = project_catalog.load("CIFAR10Model")

#     loaders_config = project_catalog.load("params:loaders")
#     dataset = project_catalog.load("CIFAR10")

#     loader = load_testset(loaders_config, dataset)

#     preds = make_predictions(task, loader)

#     assert len(preds) == 2
#     assert len(preds[0]) == len(preds[1])
#     assert type(preds[0]) == type(preds[1])


def test_get_confusion_matrix():
    """Test checking get_confusion_matrix method from model_evaluation pipeline"""
    pred = torch.Tensor([1, 3, 2, 4, 3, 2, 3, 4])
    target = torch.Tensor([1, 3, 3, 5, 7, 3, 1, 4])

    matrix = get_confusion_matrix(prediction=pred, target=target)
    assert type(matrix) == torch.Tensor
    assert matrix.shape[0] == matrix.shape[1]


def test_get_metric_value():
    """Test checking get_metric_value method from model_evaluation pipeline"""
    pred = torch.Tensor([1, 3, 2, 4, 3, 2, 3, 4])
    target = torch.Tensor([1, 3, 3, 5, 7, 3, 1, 4])
    for n in ["Precision", "Recall", "Accuracy"]:
        val = get_metric_value(prediction=pred, target=target, metric_name=n)

    assert val


def test_calc_metrics(project_context: KedroContext):
    """Test checking calc_metrix method from model_evaluation pipeline"""
    project_catalog = project_context.catalog

    metrics = project_catalog.load("params:evaluation_metrics")

    pred = torch.Tensor([1, 3, 2, 4, 3, 2, 3, 4])
    target = torch.Tensor([1, 3, 3, 5, 7, 3, 1, 4])

    with pytest.raises(ValueError):
        calc_metrics(pred, target, metrics)
