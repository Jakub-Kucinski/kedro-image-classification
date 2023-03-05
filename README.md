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
To check files you can run:
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

# Dataset download

```shell
# from torchvision
kedro run --pipeline data_download
```

```shell
# from AWS S3 bucket
kedro run --pipeline data_download --params=download_options.source:aws
```