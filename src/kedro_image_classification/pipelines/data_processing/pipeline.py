from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_dataset,
                inputs=["params:loaders", "CIFAR10"],
                outputs=["train_loader", "test_loader"],
                name="load_dataset",
            )
        ]
    )
