import torch
import torchvision


RESNET_MODELS = {
    "18": (torchvision.models.resnet18, 512),
    "34": (torchvision.models.resnet34, 512),
    "50": (torchvision.models.resnet50, 2048),
    "101": (torchvision.models.resnet101, 2048)
}


class ResNet(torch.nn.Module):
    def __init__(self, resnet_size: str = "18", n_classes=10):
        super().__init__()

        base_model_fn, feat_size = RESNET_MODELS[resnet_size]
        base_model = base_model_fn(weights="IMAGENET1K_V1", progress=True)
        base_model.fc = torch.nn.Identity()  # type: ignore
        self.feature_extractor = base_model
        self.classifier = torch.nn.Linear(feat_size, n_classes)

    def forward(self, x):
        z = self.feature_extractor(x)
        return self.classifier(z), z


class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, depth=5, n_classes=10):
        super().__init__()

        if depth == 5:
            self.feature_extractor = torch.nn.Sequential(
                torch.nn.Linear(784, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU()
            )
        elif depth == 3:
            self.feature_extractor = torch.nn.Sequential(
                torch.nn.Linear(784, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
            )
        self.classifier = torch.nn.Linear(128, n_classes)

    def forward(self, x):
        z = self.feature_extractor(x)
        return self.classifier(z), z
