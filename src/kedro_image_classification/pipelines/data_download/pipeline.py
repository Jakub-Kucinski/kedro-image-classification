from kedro.pipeline import Pipeline, node, pipeline

from .nodes import dummy_download


def create_pipeline(**kwargs) -> Pipeline:
    """Function creating Data Download pipeline.
    Consists of one node performing dummy download, therefore takes no inputs
    and outputs empty CIFAR10 dataset.

    Returns:
        Pipeline: Created data_download pipeline.
    """
    return pipeline(
        [node(dummy_download, inputs=None, outputs="CIFAR10", name="dummy_download")]
    )
