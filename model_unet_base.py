import torch

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False):
        super(Conv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm1 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
        self.elu1 = torch.nn.ELU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
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
    def __init__(self, in_channels, out_channels, use_batch_norm=False, num_atrous_blocks=4):
        super(AtrousConv, self).__init__()

        self.atrous_convs = torch.nn.ModuleList()
        for k in range(0, num_atrous_blocks):
            self.atrous_convs.append(torch.nn.Conv2d(in_channels, out_channels, 3, bias=False, padding="same", padding_mode="replicate", dilation=(1 + 4 * k)))

        if use_batch_norm:
            self.batchnorm_atrous = torch.nn.GroupNorm(num_groups=out_channels * num_atrous_blocks, num_channels=out_channels * num_atrous_blocks) # instance norm
        self.elu_atrous = torch.nn.ELU(inplace=True)

        self.conv_project = torch.nn.Conv2d(out_channels * num_atrous_blocks, out_channels, 1, bias=False, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm_project = torch.nn.GroupNorm(num_groups=out_channels, num_channels=out_channels) # instance norm
        self.elu_project = torch.nn.ELU(inplace=True)

        self.use_batch_norm = use_batch_norm
        self.num_atrous_blocks = num_atrous_blocks

    def forward(self, x):
        x = torch.cat([self.atrous_convs[k](x) for k in range(self.num_atrous_blocks)], dim=1)
        if self.use_batch_norm:
            x = self.batchnorm_atrous(x)
        x = self.elu_atrous(x)
        x = self.conv_project(x)
        if self.use_batch_norm:
            x = self.batchnorm_project(x)
        x = self.elu_project(x)
        return x

class ResConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, downsample=False, bottleneck_expansion=1, squeeze_excitation=False):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels without bottleneck expansion. The actual number of output channels is out_channels * bottleneck_expansion
        :param use_batch_norm: whether to use batch (instance) normalization
        :param downsample: whether to downsample the input 2x2
        :param bottleneck_expansion: the expansion factor of the bottleneck
        """
        super(ResConvBlock, self).__init__()
        assert in_channels <= out_channels * bottleneck_expansion

        if downsample:
            self.avgpool = torch.nn.AvgPool2d(2)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm1 = torch.nn.GroupNorm(num_groups=out_channels, num_channels=out_channels) # instance norm
        self.elu1 = torch.nn.ELU(inplace=True)

        if downsample:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, bias=False, padding=0, groups=out_channels // 8) # x8d, meaning 8 channels in each "capacity" connection
        else:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding="same", padding_mode="replicate", groups=out_channels // 8)
        if use_batch_norm:
            self.batchnorm2 = torch.nn.GroupNorm(num_groups=out_channels, num_channels=out_channels) # instance norm
        self.elu2 = torch.nn.ELU(inplace=True)

        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * bottleneck_expansion, 1, bias=False, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm3 = torch.nn.GroupNorm(num_groups=out_channels * bottleneck_expansion, num_channels=out_channels * bottleneck_expansion) # instance norm
        self.elu3 = torch.nn.ELU(inplace=True)

        if squeeze_excitation:
            self.se_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.se_conv1 = torch.nn.Conv2d(out_channels * bottleneck_expansion, out_channels // 4, 1, bias=True, padding="same", padding_mode="replicate")
            self.se_relu = torch.nn.ReLU(inplace=True)
            self.se_conv2 = torch.nn.Conv2d(out_channels // 4, out_channels * bottleneck_expansion, 1, bias=True, padding="same", padding_mode="replicate")
            self.se_sigmoid = torch.nn.Sigmoid()

        self.use_batch_norm = use_batch_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.bottleneck_expansion = bottleneck_expansion
        self.squeeze_excitation = squeeze_excitation

    def forward(self, x):
        if self.in_channels < self.out_channels * self.bottleneck_expansion:
            x_init = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.out_channels * self.bottleneck_expansion - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = self.elu1(x)

        if self.downsample:
            x = self.conv2(torch.nn.functional.pad(x, (1, 0, 1, 0), "reflect"))
        else:
            x = self.conv2(x)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = self.elu2(x)

        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.batchnorm3(x)

        if self.squeeze_excitation:
            x_se = self.se_pool(x)
            x_se = self.se_conv1(x_se)
            x_se = self.se_relu(x_se)
            x_se = self.se_conv2(x_se)
            x_se = self.se_sigmoid(x_se)
            x = x * x_se

        if self.downsample:
            x_init = self.avgpool(x_init)
        result = self.elu3(x_init + x)
        return result

class ResConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, downsample=False, blocks=3, bottleneck_expansion=1, squeeze_excitation=False):
        # bottleneck expansion means how many times the number of channels is increased in the ultimate outputs of resconvs.
        super(ResConv, self).__init__()

        self.conv_res = torch.nn.ModuleList()
        self.conv_res.append(ResConvBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, downsample=downsample, bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation))
        for k in range(1, blocks):
            self.conv_res.append(ResConvBlock(out_channels * bottleneck_expansion, out_channels, use_batch_norm=use_batch_norm, bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation))

        self.blocks = blocks
    def forward(self, x):
        for k in range(self.blocks):
            x = self.conv_res[k](x)

        return x

class UNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, res_conv_blocks=[2, 3, 4, 6, 10, 15, 15], bottleneck_expansion=1, squeeze_excitation=False):
        super(UNetBackbone, self).__init__()
        self.pyr_height = pyr_height
        self.conv_down = torch.nn.ModuleList()

        self.initial_conv = torch.nn.Conv2d(in_channels, hidden_channels * bottleneck_expansion, kernel_size=7, bias=False, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.initial_batch_norm = torch.nn.GroupNorm(num_groups=hidden_channels * bottleneck_expansion, num_channels=hidden_channels * bottleneck_expansion)  # instance norm
        self.initial_elu = torch.nn.ELU(inplace=True)

        if use_res_conv:
            self.conv0 = ResConv(hidden_channels * bottleneck_expansion, hidden_channels, use_batch_norm=use_batch_norm, blocks=res_conv_blocks[0], bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation)
            for i in range(pyr_height - 1):
                self.conv_down.append(ResConv(bottleneck_expansion * hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm, downsample=True, blocks=res_conv_blocks[i + 1], bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation))
        else:
            self.conv0 = Conv(hidden_channels, hidden_channels, use_batch_norm=use_batch_norm)
            for i in range(pyr_height - 1):
                self.conv_down.append(Conv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm))
        if use_res_conv:
            self.conv_down.append(ResConv(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - 1), hidden_channels * 2 ** pyr_height, use_batch_norm=use_batch_norm, downsample=True, blocks=res_conv_blocks[pyr_height], bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation))
        else:
            self.conv_down.append(Conv(hidden_channels * 2 ** (pyr_height - 1), hidden_channels * 2 ** pyr_height, use_batch_norm=use_batch_norm))
        self.maxpool = torch.nn.MaxPool2d(2)
        self.use_res_conv = use_res_conv
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.initial_conv(x)
        if self.use_batch_norm:
            x = self.initial_batch_norm(x)
        x = self.initial_elu(x)
        
        # contracting path
        ret = []
        x = self.conv0(x)
        ret.append(x)
        for i in range(self.pyr_height - 1):
            if self.use_res_conv:
                x = self.conv_down[i](x)
            else:
                x = self.conv_down[i](self.maxpool(x))
            ret.append(x)
        if not self.use_res_conv:
            x = self.conv_down[-1](self.maxpool(x))
        else:
            x = self.conv_down[-1](x)
        ret.append(x)

        return ret

class UNetEndClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False, use_atrous_conv=False, pyr_height=4, deep_supervision=False, num_classes=1, num_deep_multiclasses=0, bottleneck_expansion=1):
        """
        Note that num_deep_multiclasses are used only if deep_supervision is True
        """
        super(UNetEndClassifier, self).__init__()
        self.pyr_height = pyr_height
        self.conv_up = torch.nn.ModuleList()
        self.conv_up_transpose = torch.nn.ModuleList()
        for i in range(pyr_height):
            if (i == pyr_height - 1) and use_atrous_conv:
                self.conv_up.append(AtrousConv(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i), bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))
            else:
                self.conv_up.append(Conv(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i), bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))

        self.maxpool = torch.nn.MaxPool2d(2)
        for i in range(pyr_height):
            self.conv_up_transpose.append(torch.nn.ConvTranspose2d(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i), bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=2, stride=2, bias=True))

        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.outconv_deep = torch.nn.ModuleList()
            for i in range(pyr_height - 1 - num_deep_multiclasses):
                self.outconv_deep.append(torch.nn.Conv2d(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), 1, 1, bias=True))
            for i in range(pyr_height - 1 - num_deep_multiclasses, pyr_height - 1):
                self.outconv_deep.append(torch.nn.Conv2d(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), num_classes + 1, 1, bias=True))

        if num_classes > 1:
            self.outconv = torch.nn.Conv2d(bottleneck_expansion * hidden_channels, num_classes + 1, 1, bias=True)
        else:
            self.outconv = torch.nn.Conv2d(bottleneck_expansion * hidden_channels, 1, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

        self.num_classes = num_classes
        self.num_deep_multiclasses = num_deep_multiclasses

        assert num_deep_multiclasses <= pyr_height - 1, "num_deep_multiclasses must be less than or equal to pyr_height - 1"
        assert (num_deep_multiclasses == 0) or (num_classes > 1), "num_classes must be greater than 1 if num_deep_multiclasses > 0"


    def forward(self, x_list):
        # contracting path
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

    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, in_channels=3, use_atrous_conv=False, deep_supervision=False, num_classes=1, num_deep_multiclasses=0, res_conv_blocks=[2, 3, 4, 6, 10, 15, 15], bottleneck_expansion=1, squeeze_excitation=False):
        super(UNetClassifier, self).__init__()
        assert (bottleneck_expansion == 1) or use_res_conv, "residual convolutions must be used if bottleneck_expansion > 1"
        assert (not squeeze_excitation) or use_res_conv, "residual convolutions must be used if squeeze_excitation is True"
        self.backbone = UNetBackbone(in_channels, hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, res_conv_blocks=res_conv_blocks, bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm, use_atrous_conv=use_atrous_conv, pyr_height=pyr_height, deep_supervision=deep_supervision, num_classes=num_classes, num_deep_multiclasses=num_deep_multiclasses, bottleneck_expansion=bottleneck_expansion)
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