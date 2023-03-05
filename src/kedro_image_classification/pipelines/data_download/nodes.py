from kedro.pipeline import node
from torchvision import datasets


def torchvision_download():
    _ = datasets.CIFAR10("data/01_raw", download=True)


def aws_download():
    pass


def cloud_download(cloud):
    if cloud == "aws":
        aws_download()


def download_data(download_options):
    supported_clouds = ["aws"]
    if download_options["torchvision_download"]:
        torchvision_download()
    else:
        if download_options["cloud"] in supported_clouds:
            cloud_download(download_options["cloud"])
        else:
            raise Exception(
                f"Selected download from cloud, but provided cloud name is not "
                f"supported.\nList of supported clouds: {supported_clouds}"
            )


download_data_node = node(download_data, inputs="params:download_options", outputs=None)
