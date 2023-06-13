import torch

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, groups=1):
        super(Conv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate", groups=groups)
        if use_batch_norm:
            self.batchnorm1 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
        self.elu1 = torch.nn.ELU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate", groups=groups)
        if use_batch_norm:
            self.batchnorm2 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
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

class AtrousConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, groups=1):
        super(AtrousConv, self).__init__()

        assert groups == 1, "Groups not supported for atrous conv block"

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm1 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
        self.elu1 = torch.nn.ELU(inplace=True)

        self.conv2 = torch.nn.ModuleList()
        for k in range(1, 5):
            self.conv2.append(torch.nn.Conv2d(out_channels, out_channels // 4, 3, bias=True, padding="same", padding_mode="replicate", dilation=k))

        if use_batch_norm:
            self.batchnorm2 = torch.nn.GroupNorm(num_groups=out_channels // 8, num_channels=out_channels)
        self.elu2 = torch.nn.ELU(inplace=True)

        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm3 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
        self.elu3 = torch.nn.ELU(inplace=True)

        torch.nn.init.constant_(self.conv1.bias, 0.0)
        for k in range(1, 5):
            torch.nn.init.constant_(self.conv2[k - 1].bias, 0.0)

        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = self.elu1(x)
        x = torch.cat([self.conv2[k](x) for k in range(4)], dim=1)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.batchnorm3(x)
        x = self.elu3(x)
        return x

class ResConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, groups=1):
        super(ResConvBlock, self).__init__()
        assert in_channels <= out_channels
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate", groups=groups)
        if use_batch_norm:
            self.batchnorm1 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
        self.elu1 = torch.nn.ELU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate", groups=groups)

        torch.nn.init.constant_(self.conv1.bias, 0.0)
        torch.nn.init.constant_(self.conv2.bias, 0.0)

        self.use_batch_norm = use_batch_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.group_size = in_channels // groups
        self.group_pad = (out_channels - in_channels) // groups

    def forward(self, x):
        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x.view(x.shape[0], self.groups, self.group_size, x.shape[2], x.shape[3]), (0, 0, 0, 0, 0, self.group_pad), "constant", 0.0)\
                .view(x.shape[0], self.out_channels, x.shape[2], x.shape[3])
        else:
            x_init = x
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = self.elu1(x)
        x = self.conv2(x) + x_init
        return x

class ResConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, groups=1):
        super(ResConv, self).__init__()
        if in_channels > out_channels:
            if use_batch_norm:
                self.conv1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate", groups=groups),
                    torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels),
                    torch.nn.ELU(inplace=True),
                )
            else:
                self.conv1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate", groups=groups),
                    torch.nn.ELU(inplace=True),
                )
        else:
            self.conv1 = ResConvBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, groups=groups)
        self.conv_res2 = ResConvBlock(out_channels, out_channels, use_batch_norm=use_batch_norm, groups=groups)
        self.conv_res3 = ResConvBlock(out_channels, out_channels, use_batch_norm=use_batch_norm, groups=groups)
        self.conv_res4 = ResConvBlock(out_channels, out_channels, use_batch_norm=use_batch_norm, groups=groups)
        self.conv_res5 = ResConvBlock(out_channels, out_channels, use_batch_norm=use_batch_norm, groups=groups)

        self.use_batch_norm = use_batch_norm
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_res2(x)
        x = self.conv_res3(x)
        x = self.conv_res4(x)
        x = self.conv_res5(x)

        return x


class UNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, use_batch_norm=False, use_res_conv=False, use_atrous_conv=False, pyr_height=4):
        super(UNetBackbone, self).__init__()
        self.pyr_height = pyr_height
        self.conv_down = torch.nn.ModuleList()
        if use_res_conv:
            self.conv0 = ResConv(in_channels, hidden_channels, use_batch_norm=use_batch_norm)
            for i in range(pyr_height - 1):
                self.conv_down.append(ResConv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm))
        else:
            self.conv0 = Conv(in_channels, hidden_channels, use_batch_norm=use_batch_norm)
            for i in range(pyr_height - 1):
                self.conv_down.append(Conv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm))
        if use_atrous_conv:
            self.conv_down.append(AtrousConv(hidden_channels * 2 ** (pyr_height - 1), hidden_channels * 2 ** pyr_height, use_batch_norm=use_batch_norm))
        elif use_res_conv:
            self.conv_down.append(ResConv(hidden_channels * 2 ** (pyr_height - 1), hidden_channels * 2 ** pyr_height, use_batch_norm=use_batch_norm))
        else:
            self.conv_down.append(Conv(hidden_channels * 2 ** (pyr_height - 1), hidden_channels * 2 ** pyr_height, use_batch_norm=use_batch_norm))
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
    def __init__(self, hidden_channels, use_batch_norm=False, pyr_height=4, deep_supervision=False, num_classes=1, num_deep_multiclasses=0):
        """
        Note that num_deep_multiclasses are used only if deep_supervision is True
        """
        super(UNetEndClassifier, self).__init__()
        self.pyr_height = pyr_height
        self.conv_up = torch.nn.ModuleList()
        self.conv_up_transpose = torch.nn.ModuleList()
        for i in range(pyr_height):
            self.conv_up.append(Conv(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))

        self.maxpool = torch.nn.MaxPool2d(2)
        for i in range(pyr_height):
            self.conv_up_transpose.append(torch.nn.ConvTranspose2d(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=2, stride=2, bias=True))

        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.outconv_deep = torch.nn.ModuleList()
            for i in range(pyr_height - 1 - num_deep_multiclasses):
                self.outconv_deep.append(torch.nn.Conv2d(hidden_channels * 2 ** (pyr_height - i - 1), 1, 1, bias=True))
            for i in range(pyr_height - 1 - num_deep_multiclasses, pyr_height - 1):
                self.outconv_deep.append(torch.nn.Conv2d(hidden_channels * 2 ** (pyr_height - i - 1), num_classes + 1, 1, bias=True))

        if num_classes > 1:
            self.outconv = torch.nn.Conv2d(hidden_channels, num_classes + 1, 1, bias=True)
        else:
            self.outconv = torch.nn.Conv2d(hidden_channels, 1, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

        self.num_classes = num_classes
        self.num_deep_multiclasses = num_deep_multiclasses

        assert num_deep_multiclasses < pyr_height - 1, "num_deep_multiclasses must be less than pyr_height - 1"
        assert (num_deep_multiclasses == 0) or (num_classes > 1), "num_classes must be greater than 1 if num_deep_multiclasses > 0"


    def forward(self, x_list):
        # expanding path
        x = self.conv_up[0](torch.concat([self.conv_up_transpose[0](x_list[self.pyr_height]), x_list[self.pyr_height - 1]], dim=1))
        if self.deep_supervision:
            deep_outputs = [torch.squeeze(self.sigmoid(self.outconv_deep[0](x)), dim=1)]
        for i in range(1, self.pyr_height):
            x = self.conv_up[i](torch.concat([self.conv_up_transpose[i](x), x_list[self.pyr_height - i - 1]], dim=1))
            if self.deep_supervision:
                if i < self.pyr_height - 1 - self.num_deep_multiclasses:
                    deep_outputs.append(torch.squeeze(self.sigmoid(self.outconv_deep[i](x)), dim=1))
                elif i < self.pyr_height - 1:
                    deep_outputs.append(self.outconv_deep[i](x))

        if self.num_classes > 1:
            result = self.outconv(x)
        else:
            result = torch.squeeze(self.sigmoid(self.outconv(x)), dim=1)
        if self.deep_supervision:
            return result, deep_outputs
        return result

class UNetClassifier(torch.nn.Module):

    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, in_channels=3, use_atrous_conv=False, deep_supervision=False, num_classes=1, num_deep_multiclasses=0):
        super(UNetClassifier, self).__init__()
        self.backbone = UNetBackbone(in_channels, hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, use_atrous_conv=use_atrous_conv)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm, pyr_height=pyr_height, deep_supervision=deep_supervision, num_classes=num_classes, num_deep_multiclasses=num_deep_multiclasses)
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