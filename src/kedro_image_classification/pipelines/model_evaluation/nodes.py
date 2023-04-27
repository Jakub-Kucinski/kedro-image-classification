# from inspect import getmembers, isclass
from typing import Literal, Optional, Tuple

import torch
import torchmetrics
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchvision.datasets.cifar import CIFAR10
from tqdm import tqdm

from kedro_image_classification.pytorch.tasks import ClassificationTask


def load_testset(
    loaders_config: dict, cifar_dataset: Tuple[CIFAR10, CIFAR10]
) -> DataLoader:
    """Function creating data loaders for train and test set
    from given loaders configuration and dataset.

    Args:
        loaders_config (dict): Dictionary with configuration of train_set
        and test_set loaders.
        cifar_dataset (Tuple[CIFAR10, CIFAR10]): Downloaded torchvision cifar dataset.

    Returns:
        DataLoader test_set data loader.
    """
    _, test_data = cifar_dataset
    test_loader = DataLoader(
        test_data,
        batch_size=loaders_config["test_loader"]["batch_size"],
        shuffle=loaders_config["test_loader"]["shuffle"],
        num_workers=loaders_config["test_loader"]["num_workers"],
    )
    return test_loader


def make_predictions(
    task: ClassificationTask, test_loader: DataLoader
) -> Tuple[Tensor, Tensor]:
    """Function making predictions on test set.

    Args:
        task (ClassificationTask): Classification task object.
        test_loader (DataLoader): Test set data loader.

    Returns:
        Tuple[Tensor, Tensor]: Tuple with predictions and targets.
    """
    target = []
    prediction = []
    task.model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            outputs = task.model(images)
            prediction.append(outputs)
            target.append(labels)
        prediction = torch.cat(prediction)
        target = torch.cat(target)
    return prediction, target


def get_confusion_matrix(
    prediction: Tensor,
    target: Tensor,
    num_classes: int = 10,
    normalize: Optional[Literal["none", "true", "pred", "all"]] = None,
) -> Tensor:
    """Function calculating confusion matrix.

    Args:
        prediction (Tensor): Tensor with predictions.
        target (Tensor): Tensor with targets.
        num_classes (int, optional): Number of classes. Defaults to 10.
        normalize (Optional[Literal["none", "true", "pred", "all"]], optional):
        Normalization mode. Defaults to None.
    """
    confusion_matrix = ConfusionMatrix(
        task="multiclass", num_classes=num_classes, normalize=normalize
    )
    return confusion_matrix(prediction, target)


def get_metric_value(
    prediction: Tensor,
    target: Tensor,
    metric_name: str,
    num_classes: int = 10,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
):
    """Function calculating value of selected metric.

    Args:
        prediction (Tensor): Tensor with predictions.
        target (Tensor): Tensor with targets.
        metric_name (str): Name of metric.
        num_classes (int, optional): Number of classes. Defaults to 10.
        average (Optional[Literal["micro", "macro", "weighted", "none"]], optional):
        Average mode. Defaults to "macro".
    """
    metric_constructor = getattr(torchmetrics, metric_name)
    metric = metric_constructor(
        task="multiclass", num_classes=num_classes, average=average
    )
    return metric(prediction, target)


def calc_metrics(prediction: Tensor, target: Tensor, evaluation_metrics: dict) -> dict:
    """Function calculating metrics.

    Args:
        prediction (Tensor): Tensor with predictions.
        target (Tensor): Tensor with targets.
        evaluation_metrics (dict): Dictionary with evaluation metrics configuration.

    Returns:
        dict: Dictionary with calculated metrics.
    """
    results = dict()
    for metric_name, parameters in evaluation_metrics.items():
        if metric_name in [
            "Precision",
            "Recall",
            "AveragePrecision",
            "Accuracy",
            "AUROC",
        ]:
            value = get_metric_value(prediction, target, metric_name, **parameters)
            results[metric_name] = value
        if metric_name == "ConfusionMatrix":
            value = get_confusion_matrix(prediction, target, **parameters)
            results[metric_name] = value
        if metric_name == "ROC":
            # TODO
            # https://torchmetrics.readthedocs.io/en/stable/classification/roc.html
            raise NotImplementedError()
        if metric_name == "PrecisionRecallCurve":
            # TODO
            # https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall_curve.html
            raise NotImplementedError()
    with open("data/08_reporting/metrics.txt", "w") as f:
        f.write(str(results))
    return results
