[![Test Download from S3](https://github.com/Jakub-Kucinski/kedro-image-classification/actions/workflows/test-action.yml/badge.svg)](https://github.com/Jakub-Kucinski/kedro-image-classification/actions/workflows/test-action.yml)

# Table of contents
- [Table of contents](#table-of-contents)
- [CIFAR-10](#cifar-10)
- [Installation](#installation)
- [Dependencies update](#dependencies-update)
- [Pre-commit installation](#pre-commit-installation)
- [AWS access configuration](#aws-access-configuration)
- [Pipelines](#pipelines)
  - [Dataset download](#dataset-download)
  - [Dataset processing](#dataset-processing)
  - [Model training](#model-training)
  - [Model evaluation](#model-evaluation)

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

## Model training
`model_training` pipeline creates model based on specified configuration in the [model_architectures.yml](conf/base/parameters/model_architectures.yml) config file and selected variant in the [model_selection.yml](conf/base/parameters/model_selection.yml). You can define your own custom pytorch model by adding its implementation to the [models.py](src/kedro_image_classification/pytorch/models.py) file.

`Pytorch Lightning Trainer` and `Pytorch Optimizer` are created according to the specification in [model_training.yml](conf/base/parameters/model_training.yml) config file.

After training, each new model version is saved as a custom `LightningCIFAR10` Kedro dataset ([pytorch_lightning.py](src/kedro_image_classification/extras/models/pytorch_lightning.py)) in the data catalog (more precisely in the [06_models](data/06_models)).

```shell
kedro run --pipeline model_training
```
Example GPU training configuration can be found under [gpu_training.yml](conf/training_confs/gpu_training/parameters/gpu_training.yml) and run by:

```shell
# from AWS S3 bucket (require previous AWS access configuration)
kedro run --pipeline data_download --env=training_confs/gpu_training
```

## Model evaluation
`model_evaluation` pipeline create the `test_loader`, makes the predictions of the latest model on the test data and calculates metrics specified in the [model_evaluation.yml](conf/base/parameters/model_evaluation.yml) config file. Versioned predictions can be found under [prediction.pkl](data/07_model_output). Calculated metrics are saved in the [data/08_reporting/metrics.txt](data/08_reporting/metrics.txt).

```shell
kedro run --pipeline model_evaluation
```

If you want to run evaluation over other model than the latest one, you need to specify a particular version when running the `model_evaluation` pipeline:
```shell
kedro run --pipeline model_evaluation --load-versions=CIFAR10Model:YYYY-MM-DDThh.mm.ss.sssZ
```
