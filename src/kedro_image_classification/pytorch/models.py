import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    """Simple ConvNet with 2 convolutional layers and 3 fully connected layers.

    Args:
        model_params (dict): Dictionary with model parameters.
    """

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomConvModel(nn.Module):
    """Custom ConvNet that can be configured with a dictionary.

    Args:
        model_config (dict): Dictionary with model parameters.
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_params = model_config["model_params"]
        layers = self.model_params["layers"]

        activations = layers["activations"]
        features_layers = []
        for layer_name in layers["features_flow"]:
            layer_type = layer_name.split("_")[0]
            layer = getattr(nn, layer_type)
            if layer_name in activations:
                features_layers.append(layer(**activations[layer_name]))
            else:
                features_layers.append(layer(**layers["features"][layer_name]))

        classifier_layers = []
        for layer_name in layers["classifier_flow"]:
            layer_type = layer_name.split("_")[0]
            layer = getattr(nn, layer_type)
            if layer_name in activations:
                classifier_layers.append(layer(**activations[layer_name]))
            else:
                classifier_layers.append(layer(**layers["classifier"][layer_name]))

        self.features = nn.Sequential(*features_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
