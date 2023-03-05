from kedro.pipeline import Pipeline, pipeline

from .nodes import download_data_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([download_data_node])
