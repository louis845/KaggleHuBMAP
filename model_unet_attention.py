import torch

import model_unet_base

# https://arxiv.org/pdf/1804.03999.pdf

class UNetEndClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, gate_activation=torch.nn.ELU(), deep_supervision=False, num_classes=1, num_deep_multiclasses=0):
        super(UNetEndClassifier, self).__init__()
        self.pyr_height = pyr_height
        self.conv_up = torch.nn.ModuleList()
        self.conv_up_transpose = torch.nn.ModuleList()
        self.gate_collection = torch.nn.ModuleList()
        self.gate_batch_norm = torch.nn.ModuleList()
        self.gate_activation = gate_activation
        self.attention_proj = torch.nn.ModuleList()
        self.attention_batch_norm = torch.nn.ModuleList()
        self.attention_upsample = torch.nn.Upsample(scale_factor=(2, 2), mode="bilinear")

        if use_res_conv:
            for i in range(pyr_height):
                self.conv_up.append(model_unet_base.ResConv(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))
        else:
            for i in range(pyr_height):
                self.conv_up.append(model_unet_base.Conv(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))

        self.maxpool = torch.nn.MaxPool2d(2)
        for i in range(pyr_height):
            self.conv_up_transpose.append(torch.nn.ConvTranspose2d(hidden_channels * 2 ** (pyr_height - i), hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=2, stride=2, bias=True))

        for i in range(pyr_height):
            self.gate_collection.append(torch.nn.Conv2d(hidden_channels * (2 ** (pyr_height - i) + 2 ** (pyr_height - i - 1)), hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=1, bias=True))
            self.gate_batch_norm.append(torch.nn.BatchNorm2d(hidden_channels * 2 ** (pyr_height - i - 1)))
            self.attention_proj.append(torch.nn.Conv2d(hidden_channels * 2 ** (pyr_height - i - 1), 1, kernel_size=1, bias=True))
            self.attention_batch_norm.append(torch.nn.BatchNorm2d(1))

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
        if self.deep_supervision:
            result, attention_layers, deep_outputs = self.compute_with_attention_layers(x_list)
            return result, deep_outputs
        return self.compute_with_attention_layers(x_list)[0]

    def compute_with_attention_layers(self, x_list):
        attention_layers = []

        # expanding path
        gate_info = self.gate_activation(self.gate_batch_norm[0](
            self.gate_collection[0](torch.concat([x_list[self.pyr_height], self.maxpool(x_list[self.pyr_height - 1])], dim=1))
        ))
        attention_layer = self.attention_upsample(self.sigmoid(
            self.attention_batch_norm[0](self.attention_proj[0](gate_info))
        ))
        attention_layers.append(attention_layer)
        x = self.conv_up[0](torch.concat(
            [self.conv_up_transpose[0](x_list[self.pyr_height]), x_list[self.pyr_height - 1] * attention_layer], dim=1))
        if self.deep_supervision:
            deep_outputs = [torch.squeeze(self.sigmoid(self.outconv_deep[0](x)), dim=1)]

        for i in range(1, self.pyr_height):
            gate_info = self.gate_activation(self.gate_batch_norm[i](
                self.gate_collection[i](torch.concat([x, self.maxpool(x_list[self.pyr_height - i - 1])], dim=1))
            ))
            attention_layer = self.attention_upsample(self.sigmoid(
                self.attention_batch_norm[i](self.attention_proj[i](gate_info)))
            )
            attention_layers.append(attention_layer)
            x = self.conv_up[i](
                torch.concat([self.conv_up_transpose[i](x), x_list[self.pyr_height - i - 1] * attention_layer], dim=1))

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
            return result, attention_layers, deep_outputs
        return result, attention_layers

class UNetClassifier(torch.nn.Module):

    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, in_channels=3, use_atrous_conv=False, deep_supervision=False, num_classes=1, num_deep_multiclasses=0):
        super(UNetClassifier, self).__init__()
        self.backbone = model_unet_base.UNetBackbone(in_channels, hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, use_atrous_conv=use_atrous_conv)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, deep_supervision=deep_supervision, num_classes=num_classes, num_deep_multiclasses=num_deep_multiclasses)
        self.pyr_height = pyr_height

    def forward(self, x):
        x_list = self.backbone(x)
        return self.classifier(x_list)

    def compute_with_attention_layers(self, x):
        x_list = self.backbone(x)
        return self.classifier.compute_with_attention_layers(x_list)