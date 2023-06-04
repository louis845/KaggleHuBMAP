import torch

import model_unet_base

class UNetEndClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, use_deep_supervision=False):
        super(UNetEndClassifier, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.sigmoid = torch.nn.Sigmoid()

        self.pyr_height = pyr_height
        self.use_deep_supervision = use_deep_supervision

        self.conv_ups = torch.nn.ModuleList()
        self.conv_up_transposes = torch.nn.ModuleList()

        if self.use_deep_supervision:
            self.outconv = torch.nn.ModuleList()
        else:
            self.outconv = torch.nn.Conv2d(hidden_channels, 1, 1, bias=True)

        for j in range(pyr_height):
            j_pyr_height = j + 1

            conv_up = torch.nn.ModuleList()
            conv_up_transpose = torch.nn.ModuleList()
            if use_res_conv:
                for i in range(j_pyr_height):
                    conv_up.append(model_unet_base.ResConv((i + 2) * hidden_channels * (2 ** (j_pyr_height - i - 1)), hidden_channels * 2 ** (j_pyr_height - i - 1), use_batch_norm=use_batch_norm))
            else:
                for i in range(j_pyr_height):
                    conv_up.append(model_unet_base.Conv((i + 2) * hidden_channels * (2 ** (j_pyr_height - i - 1)), hidden_channels * 2 ** (j_pyr_height - i - 1), use_batch_norm=use_batch_norm))

            for i in range(j_pyr_height):
                conv_up_transpose.append(torch.nn.ConvTranspose2d(hidden_channels * 2 ** (j_pyr_height - i), hidden_channels * 2 ** (j_pyr_height - i - 1), kernel_size=2, stride=2, bias=True))

            if self.use_deep_supervision:
                self.outconv.append(torch.nn.Conv2d(hidden_channels, 1, 1, bias=True))

            self.conv_ups.append(conv_up)
            self.conv_up_transposes.append(conv_up_transpose)


    def forward(self, x_list):
        # expanding path
        prev_outs = []
        for j in range(self.pyr_height + 1):
            prev_outs.append([x_list[j]])

        if self.use_deep_supervision:
            outs = []

        for j in range(self.pyr_height):
            j_pyr_height = j + 1

            pv_list = [self.conv_up_transposes[j][0](x_list[j_pyr_height])] + prev_outs[j_pyr_height - 1]
            x = self.conv_ups[j][0](torch.concat(pv_list, dim=1))
            prev_outs[j].append(x)
            for i in range(1, j_pyr_height):
                pv_list = [self.conv_up_transposes[j][i](x)] + prev_outs[j_pyr_height - i - 1]
                x = self.conv_ups[j][i](torch.concat(pv_list, dim=1))
                prev_outs[j - i].append(x)

            if self.use_deep_supervision:
                outs.append(torch.squeeze(self.sigmoid(self.outconv[j](x)), dim=1))

        if self.use_deep_supervision:
            return outs

        return torch.squeeze(self.sigmoid(self.outconv(x)), dim=1)

class UNetClassifier(torch.nn.Module):

    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, use_deep_supervision=False, in_channels=3):
        super(UNetClassifier, self).__init__()
        self.backbone = model_unet_base.UNetBackbone(in_channels, hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, use_deep_supervision=use_deep_supervision)
        self.pyr_height = pyr_height

    def forward(self, x):
        x_list = self.backbone(x)
        return self.classifier(x_list)
