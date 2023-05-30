import torch

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False):
        super(Conv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)
        self.elu1 = torch.nn.ELU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)
        self.elu2 = torch.nn.ELU(inplace=True)

        torch.nn.init.constant_(self.conv1.bias, 0.0)
        torch.nn.init.constant_(self.conv2.bias, 0.0)

        self.use_batch_norm = use_batch_norm
    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = self.elu2(x)
        return x

class UNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, use_batch_norm=False):
        super(UNetBackbone, self).__init__()
        self.conv0 = Conv(in_channels, hidden_channels, use_batch_norm=use_batch_norm)
        self.conv1 = Conv(hidden_channels, hidden_channels * 2, use_batch_norm=use_batch_norm)
        self.conv2 = Conv(hidden_channels * 2, hidden_channels * 4, use_batch_norm=use_batch_norm)
        self.conv3 = Conv(hidden_channels * 4, hidden_channels * 8, use_batch_norm=use_batch_norm)
        self.conv4 = Conv(hidden_channels * 8, hidden_channels * 16, use_batch_norm=use_batch_norm)
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        # contracting path
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x = self.conv4(self.maxpool(x3))

        return x0, x1, x2, x3, x

class UNetEndClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False):
        super(UNetEndClassifier, self).__init__()
        self.conv1 = Conv(hidden_channels * 16, hidden_channels * 8, use_batch_norm=use_batch_norm)
        self.conv2 = Conv(hidden_channels * 8, hidden_channels * 4, use_batch_norm=use_batch_norm)
        self.conv3 = Conv(hidden_channels * 4, hidden_channels * 2, use_batch_norm=use_batch_norm)
        self.conv4 = Conv(hidden_channels * 2, hidden_channels, use_batch_norm=use_batch_norm)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upconv1 = torch.nn.ConvTranspose2d(hidden_channels * 16, hidden_channels * 8, kernel_size=2, stride=2, bias=True)
        self.upconv2 = torch.nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=2, stride=2, bias=True)
        self.upconv3 = torch.nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=2, stride=2, bias=True)
        self.upconv4 = torch.nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2, bias=True)
        self.outconv = torch.nn.Conv2d(hidden_channels, 1, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x0, x1, x2, x3, x4):
        # expanding path
        x = self.conv1(torch.concat([self.upconv1(x4), x3], dim=1))
        x = self.conv2(torch.concat([self.upconv2(x), x2], dim=1))
        x = self.conv3(torch.concat([self.upconv3(x), x1], dim=1))
        x = self.conv4(torch.concat([self.upconv4(x), x0], dim=1))

        return torch.squeeze(self.sigmoid(self.outconv(x)), dim=1)

class UNetClassifier(torch.nn.Module):

    def __init__(self, hidden_channels, use_batch_norm=False):
        super(UNetClassifier, self).__init__()
        self.backbone = UNetBackbone(3, hidden_channels, use_batch_norm=use_batch_norm)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.backbone(x)
        return self.classifier(x0, x1, x2, x3, x4)

class UNetEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels=3, use_batch_norm=False):
        super(UNetEncoder, self).__init__()
        self.backbone = UNetBackbone(3, hidden_channels, use_batch_norm=use_batch_norm)
        self.contractor0 = torch.nn.Conv2d(hidden_channels, in_channels, 1, bias=True)
        self.contractor1 = torch.nn.Conv2d(hidden_channels * 2, in_channels, 1, bias=True)
        self.contractor2 = torch.nn.Conv2d(hidden_channels * 4, in_channels, 1, bias=True)
        self.contractor3 = torch.nn.Conv2d(hidden_channels * 8, in_channels, 1, bias=True)
        self.contractor4 = torch.nn.Conv2d(hidden_channels * 16, in_channels, 1, bias=True)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.backbone(x)
        return self.contractor0(x0), self.contractor1(x1), self.contractor2(x2), self.contractor3(x3), self.contractor4(x4)

class UNetDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels=3, use_batch_norm=False):
        super(UNetDecoder, self).__init__()
        self.upconv1 = torch.nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2, bias=True)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2, bias=True)
        self.upconv3 = torch.nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2, bias=True)
        self.upconv4 = torch.nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2, bias=True)

        self.conv1 = Conv(hidden_channels + in_channels, in_channels, use_batch_norm=use_batch_norm)
        self.conv2 = Conv(hidden_channels + in_channels, in_channels, use_batch_norm=use_batch_norm)
        self.conv3 = Conv(hidden_channels + in_channels, in_channels, use_batch_norm=use_batch_norm)
        self.conv4 = Conv(hidden_channels + in_channels, in_channels, use_batch_norm=use_batch_norm)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x0, x1, x2, x3, x4):
        x = self.conv1(torch.concat([self.upconv1(x4), x3], dim=1))
        x = self.conv2(torch.concat([self.upconv2(x), x2], dim=1))
        x = self.conv3(torch.concat([self.upconv3(x), x1], dim=1))
        x = self.conv4(torch.concat([self.upconv4(x), x0], dim=1))

        return self.sigmoid(x)