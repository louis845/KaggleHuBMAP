import os
import gc

import numpy as np
import torch

import config
import model_unet_base
import model_unet_attention

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

device_cpu = torch.device("cpu")

def copy_optimizer(model: torch.nn.Module, optimizer_from: torch.optim.SGD, optimizer_to: torch.optim.SGD, to_device):
    for param in model.parameters():
        optimizer_to.state[param]["momentum_buffer"] = optimizer_from.state[param]["momentum_buffer"].to(to_device)

def compute_weighted_means(foreground_weight, background_weight, j, length):
    return (foreground_weight * (length - 1 - j) + 0.5 * j) / (length - 1), (background_weight * (length - 1 - j) + 0.5 * j) / (length - 1)

def compute_total_weights(foreground_weight, background_weight, outputs):
    foreground_weights = []
    background_weights = []

    for j in range(0, outputs):
        foreground_weight_j, background_weight_j = compute_weighted_means(foreground_weight, background_weight, j, outputs)
        foreground_weights.append(foreground_weight_j)
        background_weights.append(background_weight_j)

    return foreground_weights, background_weights


class UNetProgressiveWrapper:

    def __init__(self, hidden_channels, training=None, use_batch_norm=False, use_res_conv=False, unet_attention=False, use_atrous_conv=False, pyr_height=4, in_channels=3, outputs=2):
        self.model = []
        self.optimizer = None
        self.outputs = outputs

        if training is not None:
            assert type(training) == float, "If training mode is chosen, it must be a float which specifies the learning rate."
            self.optimizer = []
            self.learning_rate = training

        if unet_attention:
            for k in range(outputs):
                self.model.append(model_unet_attention.UNetClassifier(hidden_channels, use_batch_norm=use_batch_norm,
                                                                      use_res_conv=use_res_conv, use_atrous_conv=use_atrous_conv, pyr_height=pyr_height, in_channels=in_channels).to(device_cpu))
                if training is not None:
                    self.optimizer.append(torch.optim.SGD(self.model[k].parameters(), lr=self.learning_rate, momentum=0.99))
        else:
            for k in range(outputs):
                self.model.append(model_unet_base.UNetClassifier(hidden_channels, use_batch_norm=use_batch_norm,
                                                 use_res_conv=use_res_conv, use_atrous_conv=use_atrous_conv, pyr_height=pyr_height, in_channels=in_channels).to(device_cpu))
                if training is not None:
                    self.optimizer.append(torch.optim.SGD(self.model[k].parameters(), lr=self.learning_rate, momentum=0.99))

        self.decay_exponent_base = 1e-1

    def compute_metrics(self, cumprod_probas, ground_truth_mask, foreground_mask, foreground_weight, background_weight, frozen_outputs=0):
        foreground_weights, background_weights = compute_total_weights(foreground_weight, background_weight, self.outputs)

        true_positive = np.zeros(self.outputs - frozen_outputs, dtype=np.int64)
        false_positive = np.zeros(self.outputs - frozen_outputs, dtype=np.int64)
        false_negative = np.zeros(self.outputs - frozen_outputs, dtype=np.int64)
        true_negative = np.zeros(self.outputs - frozen_outputs, dtype=np.int64)
        loss = np.zeros(self.outputs - frozen_outputs, dtype=np.float64)
        cum_loss = 0.0

        with torch.no_grad():
            predict = cumprod_probas > 0.5
            not_predict = ~predict

            for k in range(frozen_outputs, self.outputs):
                true_positive[k - frozen_outputs] = torch.sum(predict[:, k, :, :] & (ground_truth_mask == 1)).item()
                false_positive[k - frozen_outputs] = torch.sum(predict[:, k, :, :] & (ground_truth_mask == 0)).item()
                false_negative[k - frozen_outputs] = torch.sum(not_predict[:, k, :, :] & (ground_truth_mask == 1)).item()
                true_negative[k - frozen_outputs] = torch.sum(not_predict[:, k, :, :] & (ground_truth_mask == 0)).item()

                k_loss = torch.nn.functional.binary_cross_entropy(cumprod_probas[:, k, :, :], ground_truth_mask, reduction="none")
                k_loss = torch.sum(k_loss * foreground_mask * foreground_weights[k] + k_loss * (1.0 - foreground_mask) * background_weights[k]).item()

                loss[k - frozen_outputs] = k_loss
                cum_loss += k_loss * np.power(self.decay_exponent_base, k - frozen_outputs)

        return true_positive, false_positive, false_negative, true_negative, loss, cum_loss


    def inference(self, x):
        probas = []
        with torch.no_grad():
            for k in range(self.outputs):
                self.model[k].to(config.device)
                probas.append(self.model[k](x))

                torch.cuda.empty_cache()
                gc.collect()

                self.model[k].to(device_cpu)

            return torch.cumprod(torch.stack(probas, dim=1), dim=1)

    def training(self, input, ground_truth_mask, foreground_mask, foreground_weight, background_weight, frozen_outputs=0, not_first_epoch=False):
        assert self.optimizer is not None, "Cannot train model without optimizer."

        decay_exponent_base = torch.tensor(self.decay_exponent_base, dtype=torch.float32, device=config.device)
        with torch.no_grad():
            decay_values = torch.pow(input=decay_exponent_base, exponent=torch.arange(0, self.outputs, dtype=torch.float32, device=config.device))

        with torch.no_grad():
            probas_list = []
            for k in range(self.outputs):
                self.model[k].to(config.device)
                probas_list.append(self.model[k](input))
                self.model[k].to(device_cpu)
            probas = torch.stack(probas_list, dim=1)
            return_metrics = self.compute_metrics(
                torch.cumprod(probas, dim=1), ground_truth_mask, foreground_mask, foreground_weight, background_weight, frozen_outputs=frozen_outputs
            )
        del probas_list[:]
        del probas_list
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()


        foreground_weights, background_weights = compute_total_weights(foreground_weight, background_weight, self.outputs)

        for k in range(frozen_outputs, self.outputs):
            self.model[k].to(config.device)
            optimizer = torch.optim.SGD(self.model[k].parameters(), lr=self.learning_rate, momentum=0.99)
            if not_first_epoch:
                copy_optimizer(self.model[k], self.optimizer[k], optimizer, config.device)
            optimizer.zero_grad()

            losses = []

            dummy_tensors = []
            with torch.no_grad():
                preprobas = None
                if k > 0:
                    preprobas = torch.prod(probas[:, 0:k, :, :], dim=1)
                    dummy_tensors.append(preprobas)

            for j in range(k, self.outputs):
                with torch.no_grad():
                    if j > k:
                        postprobas = torch.prod(probas[:, (k + 1):(j + 1), :, :], dim=1)
                        dummy_tensors.append(postprobas)
                        if k > 0:
                            prepostprobas = preprobas * postprobas
                            dummy_tensors.append(prepostprobas)
                        else:
                            prepostprobas = postprobas
                            dummy_tensors.append(prepostprobas)
                    else:
                        prepostprobas = preprobas
                        dummy_tensors.append(prepostprobas)

                if prepostprobas is None:
                    loss = torch.nn.functional.binary_cross_entropy(self.model[k](input), ground_truth_mask, reduction='none')
                else:
                    loss = torch.nn.functional.binary_cross_entropy(self.model[k](input) * prepostprobas, ground_truth_mask, reduction='none')
                dummy_tensors.append(loss)
                loss = torch.sum(loss * foreground_mask * foreground_weights[j] + loss * (1.0 - foreground_mask) * background_weights[j])
                losses.append(loss)
            loss = torch.sum(torch.stack(losses) * decay_values[(k - frozen_outputs):])

            loss.backward()
            optimizer.step()

            self.model[k].to(device_cpu)
            copy_optimizer(self.model[k], optimizer, self.optimizer[k], device_cpu)
            del losses[:], dummy_tensors[:]
            del optimizer, loss, losses, dummy_tensors

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

        del probas

        return return_metrics

    def save_model(self, model_dir, epoch=None):
        for k in range(self.outputs):
            if epoch is None:
                torch.save(self.model[k].state_dict(), os.path.join(model_dir, "model{}.pth".format(k)))
                if self.optimizer is not None:
                    torch.save(self.optimizer[k].state_dict(), os.path.join(model_dir, "optimizer{}.pth".format(k)))
            else:
                torch.save(self.model[k].state_dict(), os.path.join(model_dir, "model{}_epoch{}.pth".format(k, epoch)))
                if self.optimizer is not None:
                    torch.save(self.optimizer[k].state_dict(), os.path.join(model_dir, "optimizer{}_epoch{}.pth".format(k, epoch)))

    def load_model(self, model_dir):
        for k in range(self.outputs):
            self.model[k].load_state_dict(torch.load(os.path.join(model_dir, "model{}.pth".format(k))))
            if self.optimizer is not None:
                self.optimizer[k].load_state_dict(torch.load(os.path.join(model_dir, "optimizer{}.pth".format(k))))

    def multiply_gradient_momentums(self):
        assert self.optimizer is not None, "Cannot multiply gradient momentums without optimizer."
        for k in range(self.outputs):
            for param in self.model[k].parameters():
                self.optimizer[k].state[param]["momentum_buffer"] = self.optimizer[k].state[param]["momentum_buffer"] / self.decay_exponent_base

    def set_learning_rates(self, learning_rate):
        for k in range(self.outputs):
            for param_group in self.optimizer[k].param_groups:
                param_group['lr'] = learning_rate