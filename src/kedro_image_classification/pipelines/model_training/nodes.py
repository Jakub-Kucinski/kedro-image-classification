from pytorch_lightning import Trainer

import kedro_image_classification.pytorch.models as models
from kedro_image_classification.pytorch.tasks import ClassificationTask


def create_model(model_config):
    try:
        model = getattr(models, model_config["model_type"])
        return model(model_config)
    except AttributeError:
        predefined_model_types = ["SimpleConvNet", "CustomConvModel"]
        raise AttributeError(
            f"Specified unsupported model type. "
            f"Predefined model types are: {predefined_model_types}. "
            f"You can add your own custom pytorch model by adding its code to the "
            f"kedro_image_classification.pytorch.models"
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
