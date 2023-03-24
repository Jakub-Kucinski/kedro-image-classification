from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model, create_task, create_trainer, train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(create_trainer, inputs="params:trainer_params", outputs="trainer"),
            node(
                create_model,
                inputs={
                    "model_architectures": "params:model_architectures",
                    "model_selection": "params:model_selection",
                },
                outputs="model",
                name="create_model",
            ),
            node(
                create_task,
                inputs=["model", "params:optimizer_params"],
                outputs="classification_task",
                name="create_task",
            ),
            node(
                train,
                inputs=[
                    "trainer",
                    "classification_task",
                    "train_loader",
                    "test_loader",
                ],
                outputs="CIFAR10Model",
                name="train",
            ),
        ]
    )
