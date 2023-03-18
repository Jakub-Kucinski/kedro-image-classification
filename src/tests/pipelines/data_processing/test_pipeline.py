from kedro.pipeline import Pipeline

from kedro_image_classification.pipelines.data_processing.pipeline import (
    create_pipeline,
)


def test_pipeline_creation():
    """Test checking data_processing pipeline creation."""
    pipeline = create_pipeline()

    assert type(pipeline) == Pipeline
    assert len(pipeline.nodes) > 0
    assert len(pipeline.all_outputs()) == 2
    assert len(pipeline.all_inputs()) == 2
