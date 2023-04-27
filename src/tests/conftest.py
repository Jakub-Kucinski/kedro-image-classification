from pathlib import Path

import pytest
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from kedro_image_classification import settings


@pytest.fixture
def config_loader():
    return ConfigLoader(
        conf_source=str(Path.cwd() / settings.CONF_SOURCE),
    )


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="kedro_image_classification",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )
