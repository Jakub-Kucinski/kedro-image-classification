# Table of contents
1. [CIFAR-10 ](#cifar-10)
2. [Installation](#installation)
3. [Dependencies update](#dependencies-update)
4. [Pre-commit installation](#pre-commit-installation)
5. [AWS access configuration](#aws-access-configuration)
6. [Pipelines](#pipelines)
   1. [Dataset download](#dataset-download)
   2. [Dataset processing](#dataset-processing)

# CIFAR-10

Team members: Jakub Kuciński, Karol Kuczmarz, Aleksander Szymański

Project description: Classification of 32x32 colour images into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

Dataset: [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

# Installation

```shell
conda env create  --file conda.yml
conda activate kedro_image_classification
poetry install
```

# Dependencies update

```shell
poetry update
```

# Pre-commit installation

```shell
pre-commit install
pre-commit autoupdate
```
To check all files without committing simply run:
```shell
pre-commit run --all-files
```

# AWS access configuration

```shell
export AWS_SHARED_CREDENTIALS_FILE="$(pwd)/conf/local/aws/credentials"
export AWS_CONFIG_FILE="$(pwd)/conf/base/aws/config"
export LOCAL_PROFILE_NAME={profile_name}
export AWS_PROFILE=$LOCAL_PROFILE_NAME
aws configure set aws_access_key_id {aws_access_key_id} --profile $LOCAL_PROFILE_NAME
aws configure set aws_secret_access_key {aws_secret_access_key} --profile $LOCAL_PROFILE_NAME
```

# Pipelines

## Dataset download

[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) can be downloaded manually or with usage of `data_download` pipeline from torchvision or AWS S3 bucket. You can change the source of the data by changing `CIFAR10: save_args: source:` option in [catalog.yml](conf/base/catalog.yml). Possible options are `torchvision` and `aws`. Then run:
```shell
# from torchvision
kedro run --pipeline data_download
```

Alternatively you can run one of the two predefined download configurations:
```shell
# from torchvision
kedro run --pipeline data_download --env=download_confs/torchvision_download
```
```shell
# from AWS S3 bucket (require previous AWS access configuration)
kedro run --pipeline data_download --env=download_confs/aws_download
```

## Dataset processing

`data_processing` pipeline requires that the data has already been downloaded (check [Dataset download](#dataset-download) section). It creates `train_loader, test_loader` based on the [data_processing.yml](conf/base/parameters/data_processing.yml) config file.

```shell
kedro run --pipeline data_processing
```
