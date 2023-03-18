import os

from torchvision import datasets


def torchvision_download():
    _ = datasets.CIFAR10("data/01_raw", download=True)


def aws_download():
    os.system("dvc pull")
    os.system("tar -xf data/01_raw/cifar-10-python.tar.gz -C data/01_raw/")
    os.system("rm data/01_raw/cifar-10-python.tar.gz")


def cloud_download(cloud):
    if cloud == "aws":
        aws_download()


def data_download(download_options):
    supported_clouds = ["aws"]
    if download_options["source"] == "torchvision":
        torchvision_download()
    else:
        if download_options["source"] in supported_clouds:
            cloud_download(download_options["source"])
        else:
            raise Exception(
                f"Selected download from cloud, but provided cloud name is not "
                f"supported.\nList of supported clouds: {supported_clouds}"
            )


def dummy_download():
    data = {}
    return data
