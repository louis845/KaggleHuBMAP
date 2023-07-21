import torch

import model_unet_base

# https://arxiv.org/pdf/1804.03999.pdf

class UNetEndClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, use_batch_norm=False, use_atrous_conv=False, atrous_outconv_split=False, atrous_outconv_residual=False, pyr_height=4, gate_activation=torch.nn.ReLU(inplace=True), deep_supervision=False, num_classes=1, num_deep_multiclasses=0, bottleneck_expansion=1):
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

        for i in range(pyr_height):
            if (i == pyr_height - 1) and use_atrous_conv:
                self.conv_up.append(model_unet_base.AtrousConv(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i), bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm, output_intermediate=atrous_outconv_split, residual=atrous_outconv_residual))
            else:
                self.conv_up.append(model_unet_base.Conv(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i), bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), use_batch_norm=use_batch_norm))

        self.maxpool = torch.nn.MaxPool2d(2)
        for i in range(pyr_height):
            self.conv_up_transpose.append(torch.nn.ConvTranspose2d(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i), bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=2, stride=2, bias=True))

        for i in range(pyr_height):
            self.gate_collection.append(torch.nn.Conv2d(bottleneck_expansion * hidden_channels * (2 ** (pyr_height - i) + 2 ** (pyr_height - i - 1)), hidden_channels * 2 ** (pyr_height - i - 1), kernel_size=1, bias=False))
            self.gate_batch_norm.append(torch.nn.GroupNorm(num_groups=hidden_channels * 2 ** (pyr_height - i - 1), num_channels=hidden_channels * 2 ** (pyr_height - i - 1)))
            self.attention_proj.append(torch.nn.Conv2d(hidden_channels * 2 ** (pyr_height - i - 1), 1, kernel_size=1, bias=False))
            self.attention_batch_norm.append(torch.nn.GroupNorm(num_groups=1, num_channels=1))

        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.outconv_deep = torch.nn.ModuleList()
            for i in range(pyr_height - 1 - num_deep_multiclasses):
                self.outconv_deep.append(torch.nn.Conv2d(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), 1, 1, bias=True))
            for i in range(pyr_height - 1 - num_deep_multiclasses, pyr_height - 1):
                self.outconv_deep.append(torch.nn.Conv2d(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - i - 1), num_classes + 1, 1, bias=True))

        outconv_in = (2 * bottleneck_expansion * hidden_channels) if use_atrous_conv else (bottleneck_expansion * hidden_channels)
        if num_classes > 1:
            if atrous_outconv_split:
                self.outconv_mid = torch.nn.Conv2d(bottleneck_expansion * hidden_channels, 1, 1, bias=True)
                if atrous_outconv_residual:
                    self.outconv = torch.nn.Conv2d(bottleneck_expansion * hidden_channels, num_classes, 1, bias=True)
                else:
                    self.outconv = torch.nn.Conv2d(outconv_in, num_classes, 1, bias=True)
            else:
                self.outconv = torch.nn.Conv2d(outconv_in, num_classes + 1, 1, bias=True)
        else:
            self.outconv = torch.nn.Conv2d(outconv_in, 1, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

        self.num_classes = num_classes
        self.num_deep_multiclasses = num_deep_multiclasses
        self.atrous_outconv_split = atrous_outconv_split

        assert num_deep_multiclasses <= pyr_height - 1, "num_deep_multiclasses must be less than or equal to pyr_height - 1"
        assert (num_deep_multiclasses == 0) or (num_classes > 1), "num_classes must be greater than 1 if num_deep_multiclasses > 0"


    def forward(self, x_list, diagnosis=False):
        if self.deep_supervision:
            if diagnosis:
                result, attention_layers, deep_outputs, diagnosis_outputs = self.compute_with_attention_layers(x_list, diagnosis=diagnosis)
                return result, attention_layers, deep_outputs, diagnosis_outputs
            result, attention_layers, deep_outputs = self.compute_with_attention_layers(x_list, diagnosis=diagnosis)
            return result, deep_outputs
        if diagnosis:
            result, attention_layers, diagnosis_outputs = self.compute_with_attention_layers(x_list, diagnosis=diagnosis)
            return result, attention_layers, diagnosis_outputs
        return self.compute_with_attention_layers(x_list, diagnosis=diagnosis)[0]

    def compute_with_attention_layers(self, x_list, diagnosis=False):
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
            if self.num_deep_multiclasses == self.pyr_height - 1:
                deep_outputs = [self.outconv_deep[0](x)]
            else:
                deep_outputs = [torch.squeeze(self.sigmoid(self.outconv_deep[0](x)), dim=1)]
        if diagnosis:
            diagnosis_outputs = [x]

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

            if diagnosis:
                diagnosis_outputs.append(x)

        if self.num_classes > 1:
            if self.atrous_outconv_split:
                x, x_mid = x
                result = torch.cat([self.outconv_mid(x_mid), self.outconv(x)], dim=1)
            else:
                result = self.outconv(x)
        else:
            result = torch.squeeze(self.sigmoid(self.outconv(x)), dim=1)
        if self.deep_supervision:
            if diagnosis:
                return result, attention_layers, deep_outputs, diagnosis_outputs
            else:
                return result, attention_layers, deep_outputs
        if diagnosis:
            return result, attention_layers, diagnosis_outputs
        return result, attention_layers

class UNetClassifier(torch.nn.Module):

    def __init__(self, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4, in_channels=3, use_atrous_conv=False, atrous_outconv_split=False, atrous_outconv_residual=False, deep_supervision=False, num_classes=1, num_deep_multiclasses=0, res_conv_blocks=[2, 3, 4, 6, 10, 15, 15], bottleneck_expansion=1, squeeze_excitation=False, use_initial_conv=False):
        super(UNetClassifier, self).__init__()
        assert (bottleneck_expansion == 1) or use_res_conv, "residual convolutions must be used if bottleneck_expansion > 1"
        assert (not squeeze_excitation) or use_res_conv, "residual convolutions must be used if squeeze_excitation is True"
        assert (not atrous_outconv_split) or (use_atrous_conv and (num_classes > 1)), "atrous_outconv_split can only be used if use_atrous_conv and num_classes > 1"
        self.backbone = model_unet_base.UNetBackbone(in_channels, hidden_channels, use_batch_norm=use_batch_norm, use_res_conv=use_res_conv, pyr_height=pyr_height, res_conv_blocks=res_conv_blocks, bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation, use_initial_conv=use_initial_conv)
        self.classifier = UNetEndClassifier(hidden_channels, use_batch_norm=use_batch_norm, use_atrous_conv=use_atrous_conv, atrous_outconv_split=atrous_outconv_split, atrous_outconv_residual=atrous_outconv_residual, pyr_height=pyr_height, deep_supervision=deep_supervision, num_classes=num_classes, num_deep_multiclasses=num_deep_multiclasses, bottleneck_expansion=bottleneck_expansion)
        self.pyr_height = pyr_height

    def forward(self, x, diagnosis=False):
        x_list = self.backbone(x)
        if diagnosis:
            return {"encoder": x_list, "classifier": self.classifier(x_list, diagnosis=diagnosis)}
        return self.classifier(x_list, diagnosis=diagnosis)

    def compute_with_attention_layers(self, x):
        x_list = self.backbone(x)
        return self.classifier.compute_with_attention_layers(x_list)