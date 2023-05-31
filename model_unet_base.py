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

class ResConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False):
        super(ResConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)
        self.elu1 = torch.nn.ELU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)
        self.elu2 = torch.nn.ELU(inplace=True)

        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm3 = torch.nn.BatchNorm2d(out_channels)
        self.elu3 = torch.nn.ELU(inplace=True)
        self.conv4 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm4 = torch.nn.BatchNorm2d(out_channels)
        self.elu4 = torch.nn.ELU(inplace=True)

        self.conv5 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm5 = torch.nn.BatchNorm2d(out_channels)
        self.elu5 = torch.nn.ELU(inplace=True)

        torch.nn.init.constant_(self.conv1.bias, 0.0)
        torch.nn.init.constant_(self.conv2.bias, 0.0)
        torch.nn.init.constant_(self.conv3.bias, 0.0)
        torch.nn.init.constant_(self.conv4.bias, 0.0)
        torch.nn.init.constant_(self.conv5.bias, 0.0)

        self.use_batch_norm = use_batch_norm
    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x1 = self.elu1(x)

        x = self.conv2(x1)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.conv3(x) + x1
        if self.use_batch_norm:
            x = self.batchnorm3(x)
        x2 = self.elu3(x)

        x = self.conv4(x2)
        if self.use_batch_norm:
            x = self.batchnorm4(x)
        x = self.elu4(x)
        x = self.conv5(x) + x2
        if self.use_batch_norm:
            x = self.batchnorm5(x)
        x = self.elu5(x)

        return x

class UNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4):
        super(UNetBackbone, self).__init__()
        self.pyr_height = pyr_height
        self.conv_down = torch.nn.ModuleList()
        if use_res_conv:
            self.conv0 = ResConv(in_channels, hidden_channels, use_batch_norm=use_batch_norm)
            for i in range(pyr_height):
                self.conv_down.append(ResConv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm))
        else:
            self.conv0 = Conv(in_channels, hidden_channels, use_batch_norm=use_batch_norm)
            for i in range(pyr_height):
                self.conv_down.append(Conv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm))
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        # contracting path
        ret = []
        x = self.conv0(x)
        ret.append(x)
        for i in range(self.pyr_height):
            x = self.conv_down[i](self.maxpool(x))
            ret.append(x)

        return ret

class UNetEndClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4):
        super(UNetEndClassifier, self).__init__()
        self.pyr_height = pyr_height
        self.conv_up = torch.nn.ModuleList()
        self.conv_up_transpose = torch.nn.ModuleList()
        if use_res_conv:
            for i in range(pyr_height):
                self.conv_up.append(ResConv(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))
        else:
            for i in range(pyr_height):
                self.conv_up.append(Conv(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))

        self.maxpool = torch.nn.MaxPool2d(2)
        for i in range(pyr_height):
            self.conv_up_transpose.append(torch.nn.ConvTranspose2d(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=2, stride=2, bias=True))
        self.outconv = torch.nn.Conv2d(hidden_channels, 1, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x_list):
        # expanding path
        x = self.conv_up[0](torch.concat([self.conv_up_transpose[0](x_list[self.pyr_height]), x_list[self.pyr_height - 1]], dim=1))
        for i in range(1, self.pyr_height):
            x = self.conv_up[i](torch.concat([self.conv_up_transpose[i](x), x_list[self.pyr_height - i - 1]], dim=1))

        return torch.squeeze(self.sigmoid(self.outconv(x)), dim=1)

class UNetClassifier(torch.nn.Module):

    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4):
        super(UNetClassifier, self).__init__()
        self.backbone = UNetBackbone(3, hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height)
        self.pyr_height = pyr_height

    def forward(self, x):
        x_list = self.backbone(x)
        return self.classifier(x_list)

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