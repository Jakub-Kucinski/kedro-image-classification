from inspect import getmembers, isclass

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import kedro_image_classification.pytorch.models as models
from kedro_image_classification.pytorch.tasks import ClassificationTask


def create_model(model_architectures: dict, model_selection: dict) -> torch.nn.Module:
    """Function creating model from predefined architectures
    and types based on the selection. Checks whether the selected model
    is in the predefined model architectures and if the model type is defined.

    Args:
        model_architectures (dict): Dictionary with all defined model architectures.
        model_selection (dict): Dictionary with the model selection.

    Raises:
        KeyError: Raised when the selected model is not in the model_architectures.
        AttributeError: Raised when type of the model from the architecture
        is not predefined.

    Returns:
        torch.nn.Module: Created model object.
    """
    try:
        model_config = model_architectures[model_selection["model_name"]]
    except KeyError:
        predefined_model_names = list(model_architectures.keys())
        raise KeyError(
            f"Selected non existing model name: '{model_selection['model_name']}'. "
            f"Predefined model names are: {predefined_model_names}."
            f"You can add your own custom model by adding its configuration"
            f"to the model_architectures.yml file."
        )

    try:
        model_constructor = getattr(models, model_config["model_type"])
    except AttributeError:
        predefined_model_types = [
            name for name, obj in getmembers(models) if isclass(obj)
        ]
        raise AttributeError(
            f"Specified unsupported model type. "
            f"Predefined model types are: {predefined_model_types}. "
            f"You can add your own custom pytorch model by adding its code to the "
            f"kedro_image_classification.pytorch.models"
        )

    return model_constructor(model_config)


def create_trainer(trainer_params: dict) -> Trainer:
    """Function creating pytorch_lightning trainer with given parameters.

    Args:
        trainer_params (dict): Dictionary with parameters for the trainer object.

    Returns:
        Trainer: Created pytorch_lightning trainer object.
    """
    trainer = Trainer(
        default_root_dir="data/06_models",
        accelerator=trainer_params["accelerator"],
        # logger=wandb_logger,
        log_every_n_steps=trainer_params["log_every_n_steps"],
        val_check_interval=trainer_params["val_check_interval"],
        devices=trainer_params["devices"],
        max_epochs=trainer_params["max_epochs"],
        # plugins=DDPPlugin(find_unused_parameters=False),
    )
    return trainer


def create_task(model: torch.nn.Module, optimizer_params: dict) -> ClassificationTask:
    """Function creating classification task (pytorch_lightning lightning module)
    from model (created in the previous pipeline part) and provided parameters
    of the optimizer.

    Args:
        model (torch.nn.Module): Model created in the create_model node.
        optimizer_params (dict): Dictionary with parameters for the optimizer.

    Returns:
        ClassificationTask: Lightning module with functions required for model training.
    """
    return ClassificationTask(model, optimizer_params)


def train(
    trainer: Trainer,
    task: ClassificationTask,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Trainer:
    """Function performing training of the model.

    Args:
        trainer (Trainer): PyTorch Lightning trainer which will perform
        the task of training.
        task (ClassificationTask): The model to train wrapped in Classification task.
        train_loader (DataLoader): Loader of the train set.
        test_loader (DataLoader): Loader of the test set.

    Returns:
        Trainer: Given trainer after the training process
    """
    trainer.fit(task, train_loader, test_loader)
    return trainer
