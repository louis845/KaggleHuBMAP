import torch

import model_unet_base

class UNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, outputs=2):
        super(UNetBackbone, self).__init__()
        self.pyr_height = pyr_height
        self.conv_down = torch.nn.ModuleList()
        self.first_conv = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        if use_res_conv:
            self.conv0 = model_unet_base.ResConv(hidden_channels, hidden_channels, use_batch_norm=use_batch_norm, groups=outputs)
            for i in range(pyr_height):
                self.conv_down.append(model_unet_base.ResConv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm, groups=outputs))
        else:
            self.conv0 = model_unet_base.Conv(hidden_channels, hidden_channels, use_batch_norm=use_batch_norm, groups=outputs)
            for i in range(pyr_height):
                self.conv_down.append(model_unet_base.Conv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm, groups=outputs))
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        # contracting path
        ret = []
        x = self.conv0(self.first_conv(x))

        ret.append(x)
        for i in range(self.pyr_height):
            x = self.conv_down[i](self.maxpool(x))
            ret.append(x)

        return ret

class UNetEndClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, outputs=2):
        super(UNetEndClassifier, self).__init__()
        self.pyr_height = pyr_height
        self.conv_up = torch.nn.ModuleList()
        self.conv_up_transpose = torch.nn.ModuleList()
        """if use_res_conv:
            for i in range(pyr_height):
                self.conv_up.append(model_unet_base.ResConv(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm, groups=outputs))
        else:"""
        for i in range(pyr_height):
            self.conv_up.append(model_unet_base.Conv(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm, groups=outputs))

        self.maxpool = torch.nn.MaxPool2d(2)
        for i in range(pyr_height):
            self.conv_up_transpose.append(torch.nn.ConvTranspose2d(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=2, stride=2, bias=True, groups=outputs))
        self.outconv = torch.nn.Conv2d(hidden_channels, out_channels=outputs, kernel_size=1, bias=True, groups=outputs)
        self.sigmoid = torch.nn.Sigmoid()

        self.outputs = outputs

    def forward(self, x_list):
        batch_size = x_list[0].shape[0]

        # expanding path
        x = self.conv_up[0](torch.concat([
                self.conv_up_transpose[0](x_list[self.pyr_height]).view(batch_size, self.outputs, x_list[self.pyr_height - 1].shape[1] // self.outputs, x_list[self.pyr_height - 1].shape[2], x_list[self.pyr_height - 1].shape[3]),
                x_list[self.pyr_height - 1].view(batch_size, self.outputs, x_list[self.pyr_height - 1].shape[1] // self.outputs, x_list[self.pyr_height - 1].shape[2], x_list[self.pyr_height - 1].shape[3])],
            dim=2).view(batch_size, -1, x_list[self.pyr_height - 1].shape[2], x_list[self.pyr_height - 1].shape[3]))
        for i in range(1, self.pyr_height):
            x = self.conv_up[i](torch.concat([
                    self.conv_up_transpose[i](x).view(batch_size, self.outputs, x_list[self.pyr_height - i - 1].shape[1] // self.outputs, x_list[self.pyr_height - i - 1].shape[2], x_list[self.pyr_height - i - 1].shape[3]),
                    x_list[self.pyr_height - i - 1].view(batch_size, self.outputs, x_list[self.pyr_height - i - 1].shape[1] // self.outputs, x_list[self.pyr_height - i - 1].shape[2], x_list[self.pyr_height - i - 1].shape[3])],
            dim=2).view(batch_size, -1, x_list[self.pyr_height - i - 1].shape[2], x_list[self.pyr_height - i - 1].shape[3]))

        probas = self.sigmoid(self.outconv(x))

        return torch.cumprod(probas, dim=1)

class UNetClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, in_channels=3, outputs=2):
        super(UNetClassifier, self).__init__()
        self.backbone = UNetBackbone(in_channels, hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, outputs=outputs)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, outputs=outputs)
        self.pyr_height = pyr_height

    def forward(self, x):
        x_list = self.backbone(x)
        return self.classifier(x_list)