from kedro.pipeline import Pipeline

from kedro_image_classification.pipelines.data_download.pipeline import create_pipeline


def test_pipeline_creation():
    """Test checking data_download pipeline creation."""
    pipeline = create_pipeline()

    assert type(pipeline) == Pipeline
    assert len(pipeline.nodes) > 0
    assert pipeline.all_outputs() == {"CIFAR10"}
    assert len(pipeline.all_inputs()) == 0
