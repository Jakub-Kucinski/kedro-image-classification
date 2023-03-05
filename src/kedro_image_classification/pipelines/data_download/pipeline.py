from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_download


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [node(data_download, inputs="params:download_options", outputs=None)]
    )
