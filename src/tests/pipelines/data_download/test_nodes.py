import os

from kedro_image_classification.pipelines.data_download.nodes import (
    aws_download,
    cloud_download,
    data_download,
    torchvision_download,
)


def nottest(obj):
    obj.__test__ = False
    return obj


@nottest
def torchvision_scenario(function, opt=None):
    files_before = os.listdir("data/01_raw")
    assert "cifar-10-batches-py" not in files_before

    if not opt:
        function()
    else:
        function(opt)

    files_after = os.listdir("data/01_raw")

    assert "cifar-10-batches-py" in files_after

    downloaded_files = os.listdir("data/01_raw/cifar-10-batches-py")
    assert len(downloaded_files) > 0

    os.system("rm -r data/01_raw/cifar-10-batches-py")


@nottest
def aws_scenario(function, opt=None):
    files_before = os.listdir("data/01_raw")
    os.system("cp data/01_raw/cifar-10-python.tar.gz data/01_raw/temp.tar.gz")

    assert "cifar-10-batches-py" not in files_before
    assert "cifar-10-python.tar.gz" in files_before

    if not opt:
        function()
    else:
        function(opt)
    files_after = os.listdir("data/01_raw")

    assert "cifar-10-batches-py" in files_after
    assert "cifar-10-python.tar.gz" not in files_after

    downloaded_files = os.listdir("data/01_raw/cifar-10-batches-py")
    assert len(downloaded_files) > 0

    os.system("rm -r data/01_raw/cifar-10-batches-py")
    os.system("mv data/01_raw/temp.tar.gz data/01_raw/cifar-10-python.tar.gz")


def test_torchvision_download():
    """Test checking data download method using torchvision datasets."""
    torchvision_scenario(torchvision_download)


def test_aws_download():
    """Test checking data download method using aws cloud."""
    aws_scenario(aws_download)


def test_cloud_download():
    """Test checking general method for data download using cloud service."""
    aws_scenario(cloud_download, "aws")


def test_data_download():
    """Test checking data_download method."""
    download_options = {"source": "gcp"}

    execption_occurred = False

    try:
        data_download(download_options)
    except Exception:
        execption_occurred = True

    assert execption_occurred

    download_options["source"] = "torchvision"

    torchvision_scenario(data_download, download_options)

    download_options["source"] = "aws"

    aws_scenario(data_download, download_options)
