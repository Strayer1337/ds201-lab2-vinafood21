import torch
import torch.nn as nn


def _pair(size):
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("input_size phải là số nguyên hoặc tuple (height, width).")
    return int(size[0]), int(size[1])


class LeNet(nn.Module):

    def __init__(self, in_ch=3, input_size=(64, 64), num_classes=21):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        height, width = _pair(input_size)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, height, width)
            n_flat = self.features(dummy).flatten(start_dim=1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_flat, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class InceptionBlock(nn.Module):
    
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3r, f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5r, f5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.out_channels = f1 + f3 + f5 + fp

    def forward(self, x):
        branches = (
            self.block1(x),
            self.block2(x),
            self.block3(x),
            self.block4(x),
        )
        return torch.cat(branches, dim=1)


class GoogLeNet(nn.Module):

    def __init__(self, in_ch=3, num_classes=21):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception1 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.pool_after2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inception3.out_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool_after2(x)
        x = self.inception3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


GoogleNet = GoogLeNet


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet18(nn.Module):

    def __init__(self, in_ch=3, num_classes=21):
        super().__init__()
        self.current_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.current_channels, out_channels, stride)]
        self.current_channels = out_channels * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.current_channels, out_channels, stride=1)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class PretrainedResNet(nn.Module):
    def __init__(
        self,
        num_classes=21,
        freeze_backbone=False,
        model_name="microsoft/resnet-50",
        pretrained=True,
        backbone_config=None,
    ):
        super().__init__()

        try:
            from transformers import ResNetConfig, ResNetForImageClassification
        except ImportError as exc:
            raise ImportError(
                "Cần cài transformers để dùng model pretrained ResNet-50."
            ) from exc

        if pretrained:
            pretrained_model = ResNetForImageClassification.from_pretrained(
                model_name
            )
            config_dict = pretrained_model.config.to_dict()
        else:
            if backbone_config is not None:
                config = ResNetConfig.from_dict(backbone_config)
            else:
                config = ResNetConfig()
            pretrained_model = ResNetForImageClassification(config)
            config_dict = config.to_dict()

        self.resnet = pretrained_model.resnet
        classifier_in_features = pretrained_model.config.hidden_sizes[-1]
        self.classifier = nn.Linear(classifier_in_features, num_classes)
        self.backbone_config = config_dict
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False
            self.resnet.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.resnet.eval()
        return self

    def forward(self, images):
        outputs = self.resnet(images)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = outputs[0]
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)

PretrainedResnet = PretrainedResNet


def build_model(
    model_name,
    num_classes,
    input_size=64,
    in_ch=3,
    freeze_backbone=False,
    pretrained=True,
    backbone_config=None,
):
    model_name = model_name.lower()

    if model_name == "lenet":
        return LeNet(
            in_ch=in_ch,
            input_size=input_size,
            num_classes=num_classes,
        )
    if model_name in ("googlenet", "google-net"):
        return GoogLeNet(in_ch=in_ch, num_classes=num_classes)
    if model_name == "resnet18":
        return ResNet18(in_ch=in_ch, num_classes=num_classes)
    if model_name in ("resnet50", "pretrained_resnet50"):
        return PretrainedResNet(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            backbone_config=backbone_config,
        )
    raise ValueError(f"Model không được hỗ trợ: {model_name}")


def n_params(module, trainable_only=False):
    parameters = module.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


