from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calc_metrics, make_predictions


def create_pipeline(**kwargs) -> Pipeline:
    """Function creating Model Evaluation pipeline creating test_loader,
    making predictions on testset and calculating metrics.

    Returns:
        Pipeline: Created Model Evaluation pipeline.
    """
    return pipeline(
        [
            node(
                make_predictions,
                inputs=["CIFAR10Model", "test_loader"],
                outputs=["Test_prediction", "Test_labels"],
                name="make_predictions",
            ),
            node(
                calc_metrics,
                inputs=["Test_prediction", "Test_labels", "params:evaluation_metrics"],
                outputs="metrics",
                name="calc_metrics",
            ),
        ],
        tags="model_evaluation",
    )
