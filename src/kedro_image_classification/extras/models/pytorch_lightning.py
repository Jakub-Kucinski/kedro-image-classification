from pathlib import PurePosixPath

import fsspec
from kedro.io import AbstractVersionedDataSet, DataSetError
from kedro.io.core import get_filepath_str, get_protocol_and_path

from kedro_image_classification.pytorch.tasks import ClassificationTask


class LightningCIFAR10(AbstractVersionedDataSet):
    """``LightningCIFAR10`` class for loading and saving PyTorch Lightning models
    in Kedro data catalog."""

    def __init__(self, path, version=None):
        protocol, path = get_protocol_and_path(path, version)
        self._protocol = protocol
        self._path = PurePosixPath(path)
        if protocol == "file":
            _fs_args = {}
            _fs_args.setdefault("auto_mkdir", True)
            self._fs = fsspec.filesystem(self._protocol, **_fs_args)
        else:
            self._fs = fsspec.filesystem(self._protocol)
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _load(self):
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        return ClassificationTask.load_from_checkpoint(load_path)

    def _save(self, lightning_module):
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        lightning_module.save_checkpoint(save_path)

    def _exists(self):
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False
        return self._fs.exists(load_path)

    def _describe(self):
        return dict(filepath=self._path, protocol=self._protocol, version=self._version)
