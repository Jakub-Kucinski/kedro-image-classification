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

# Dataset preparation

## AWS access configured
```shell
dvc pull
```