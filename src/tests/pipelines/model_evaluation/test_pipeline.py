from kedro.pipeline import Pipeline

from kedro_image_classification.pipelines.model_evaluation.pipeline import (
    create_pipeline,
)


class TestPipeline:
    def test_pipeline_creation(self):
        """Test checking data_processing pipeline creation."""

        pipeline = create_pipeline()
        assert type(pipeline) == Pipeline
        assert len(pipeline.nodes) == 3
