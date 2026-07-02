import torch
import torch.nn as nn

##defining the inception module
class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        pool_proj
    ):
        super().__init__()

        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1), # Added BatchNorm
            nn.ReLU(inplace=True)
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.BatchNorm2d(red_3x3), # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3), # Added BatchNorm
            nn.ReLU(inplace=True)
        )

        # Branch 3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.BatchNorm2d(red_5x5), # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5), # Added BatchNorm
            nn.ReLU(inplace=True)
        )

        # Branch 4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj), # Added BatchNorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
        

##defining the general Googlenet architecture
class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm2d(64), # Added BatchNorm
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.Conv2d(
                64,
                64,
                kernel_size=1
            ),
            nn.BatchNorm2d(64), # Added BatchNorm
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                192,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(192), # Added BatchNorm
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        # Inception 3
        self.inception3a = InceptionModule(
            192, 64, 96, 128, 16, 32, 32)

        self.inception3b = InceptionModule(
            256, 128, 128, 192, 32, 96, 64
        )

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Inception 4
        self.inception4a = InceptionModule(
            480, 192, 96, 208, 16, 48, 64
        )

        self.inception4b = InceptionModule(
            512, 160, 112, 224, 24, 64, 64
        )

        self.inception4c = InceptionModule(
            512, 128, 128, 256, 24, 64, 64
        )

        self.inception4d = InceptionModule(
            512, 112, 144, 288, 32, 64, 64
        )

        self.inception4e = InceptionModule(
            528, 256, 160, 320, 32, 128, 128
        )

        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Inception 5
        self.inception5a = InceptionModule(
            832, 256, 160, 320, 32, 128, 128
        )

        self.inception5b = InceptionModule(
            832, 384, 192, 384, 48, 128, 128
        )

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.stem(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.maxpool1(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.maxpool2(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        return x
