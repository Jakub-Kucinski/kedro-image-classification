from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_dataset


def create_pipeline(**kwargs) -> Pipeline:
    """Function creating Data Processing pipeline. Consists of one node
    which takes downloaded dataset and provides data loaders
    for train and test set.

    Returns:
        Pipeline: Created data_processing pipeline.
    """
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
