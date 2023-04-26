"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""
from pathlib import Path

from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner
from torch.utils.data import Subset

from kedro_image_classification.pipelines.data_download.nodes import dummy_download
from kedro_image_classification.pipelines.data_processing.nodes import load_dataset


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()


class TestModelTrainingPipeline:
    def test_something(
        self, config_loader: ConfigLoader, project_context: KedroContext
    ):
        print(project_context.params)
        print(project_context.catalog)

        # add to catalog train_loader
        # add to catalog test_loader
        # create catalog
        # catalog = project_context.catalog

        project_catalog = project_context.catalog

        project_catalog.save("CIFAR10", data=dummy_download())

        cifar_dataset = project_catalog.load("CIFAR10")

        train_loader, test_loader = load_dataset(
            loaders_config=project_catalog.load("params:loaders"),
            cifar_dataset=(
                Subset(cifar_dataset[0], indices=list(range(10))),
                Subset(cifar_dataset[1], indices=list(range(10))),
            ),
        )

        catalog = DataCatalog(
            {
                "train_loader": MemoryDataSet(data=train_loader),
                "test_loader": MemoryDataSet(data=test_loader),
                "CIFAR10Model": MemoryDataSet(data=dict(), copy_mode="assign"),
                **{
                    name: MemoryDataSet(data=project_catalog.load(name))
                    for name in project_catalog.list(regex_search="^params:")
                },
            }
        )

        # create pipeline
        from kedro_image_classification.pipelines.model_training import create_pipeline

        pipeline = create_pipeline()
        # pipeline = pipeline.to_nodes("create_task")

        # run pipeline
        runner = SequentialRunner()
        runner.run(
            pipeline=pipeline,
            catalog=catalog,
        )
        model = catalog.load("CIFAR10Model")

        # Test model
        print(model)
        assert True
