## 29.03.2023 Model evaluation and training
- Update data catolog and add model_evaluation to conf/base/parameters
- Add entry source code for model_evaluation pipeline
- Remove bad trained-model path
- Updated .gitignore, pyproject.toml and README
- Add trained dvc model
- Update trained model
- Update .gitignore
- Add trained model
- Merge pull request #4 from Jakub-Kucinski/fixes1
- added docstrings to nodes and pipelines functions
- added name attribute to nodes in pipelines for --to-nodes, --from-nodes and --node kedro run options
- model_training: fixed create_model node and the pipeline architectures
- model_training: added yml config files for the correct create_model node architecture
- model_training: removed redundant yml config files and directories
- model_training: added yml config files for the correct create_model node architecture
- model_training: removed redundant yml config files and directories
- data_download: changed tests to match the current data_download pipeline nodes
- data_download: removed legacy code


## 15.03.2023 - Model training
- data_download: added documentation to some methods
- model_training: added directory with different training configuration
- data_download: update test to the current pipeline architecture
- Add custom model confs
- Add CustomConvModel torch model and updated model_training pipeline
- Add model_training pipeline and custom LightningCIFAR10 kedro dataset
- Add custom pytorch and pytorchlightning models, model.yml and model_training.yml confs
- Add check-toml to pre-commit
- Add custom envs for torchvision and aws data_download, updated README
- Moved data downloading logic from data_download nodes into TorchvisionCIFAR10 _save method, updated data_download.yml and catalog.yml
- Updated pre-commit

## 08.03.2023 - Data collection + preparation
- data_download: implemented tests for the pipeline and nodes
- data_processing: changed tests structure to fit the kedro style
- Refactored README.md and added Dataset processing section
- Add custom TorchvisionCIFAR10 dataset and data_processing pipeline
- Refactored data_dwonload params
- Add aws_download method in pipeline
- Modify aws setup manual
- Fix unwanted changes
- Add torchvision cifar10 download
- Fix manual for aws setup in README
- Add aws local setup instruction
- Updated README
- Update gitignore
- Add torchvision and config for dvc
- Add dvc file with dataset
- Initialize DVC
- Add pre-commit
