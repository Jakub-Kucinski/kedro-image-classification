from pytorch_lightning import Trainer

from kedro_image_classification.pytorch.models import CustomConvModel, SimpleConvNet
from kedro_image_classification.pytorch.tasks import ClassificationTask


def create_model(model_config):
    supported_model_types = ["SimpleConvNet", "CustomConvModel"]
    if model_config["model_type"].lower() == "SimpleConvNet".lower():
        return SimpleConvNet(model_config)
    elif model_config["model_type"].lower() == "CustomConvModel".lower():
        return CustomConvModel(model_config)
    else:
        raise Exception(
            f"Specified unsupported model type.\n"
            f"Currently supported model types: {supported_model_types}"
        )


def create_trainer(trainer_params):
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


def create_task(model, optimizer_params):
    return ClassificationTask(model, optimizer_params)


def train(trainer, task, train_loader, test_loader):
    trainer.fit(task, train_loader, test_loader)
    return trainer
