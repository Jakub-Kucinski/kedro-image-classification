from kedro.pipeline import Pipeline

from kedro_image_classification.pipelines.model_training.pipeline import create_pipeline


class TestPipeline:
    def test_pipeline_creation(self):
        """Test checking data_processing pipeline creation."""

        pipeline = create_pipeline()
        assert type(pipeline) == Pipeline
        assert len(pipeline.nodes) == 4
        assert len(pipeline.all_outputs()) == len(pipeline.nodes)
        assert len(pipeline.all_inputs()) == 9
