from kedro.pipeline import Pipeline, node, pipeline

from .nodes import dummy_download


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(dummy_download, inputs=None, outputs="CIFAR10")])
