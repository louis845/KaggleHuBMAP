"""Obtain the raw logits from the model. The model here is trained with reconstructed_model_progressive_supervised_unet.py"""

import gc
import os
import time
import argparse

import tqdm
import torch
import torch.nn
import h5py
import numpy as np

import model_unet_base
import model_unet_attention
import inference_reconstructed_base
import config

def get_result_logits(model: torch.nn.Module, inference_batch: torch.Tensor, test_time_augmentation: bool=True):
    # inference_batch is a batch of images of shape (1, C, H, W)
    if test_time_augmentation:
        # obtain all variants using TTA
        result = get_result_logits(model, inference_batch, False)
        gc.collect()
        torch.cuda.empty_cache()
        result_rot90 = torch.rot90(get_result_logits(model, torch.rot90(inference_batch, 1, [2, 3]), False),
                                   -1, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot180 = torch.rot90(get_result_logits(model, torch.rot90(inference_batch, 2, [2, 3]), False),
                                    -2, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot270 = torch.rot90(get_result_logits(model, torch.rot90(inference_batch, 3, [2, 3]), False),
                                    -3, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        # flip
        result_flip = torch.flip(get_result_logits(model, torch.flip(inference_batch, [-1]), False), [-1])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot90_flip = torch.rot90(
            torch.flip(get_result_logits(model, torch.flip(torch.rot90(inference_batch, 1, [2, 3]), [-1]), False),
                       [-1]), -1, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot180_flip = torch.rot90(
            torch.flip(get_result_logits(model, torch.flip(torch.rot90(inference_batch, 2, [2, 3]), [-1]), False),
                       [-1]), -2, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot270_flip = torch.rot90(
            torch.flip(get_result_logits(model, torch.flip(torch.rot90(inference_batch, 3, [2, 3]), [-1]), False),
                       [-1]), -3, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()

        # now its a shape of (8, 3, H, W) tensor
        result = torch.cat([result, result_rot90, result_rot180, result_rot270, result_flip,
                              result_rot90_flip, result_rot180_flip, result_rot270_flip], dim=0)
        # permute to obtain (H, W, 3, 8) tensor
        return result.permute(2, 3, 1, 0)
    else:
        result, deep_outputs = model(inference_batch)
        del deep_outputs[:]
        # restrict result to center 512x512
        result = result[:, :, image_radius - 256:image_radius + 256, image_radius - 256:image_radius + 256]
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a multiclass reconstructed U-Net model")
    parser.add_argument("--use_batch_norm", action="store_true",
                        help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true",
                        help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true",
                        help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--use_squeeze_excitation", action="store_true",
                        help="Whether to use squeeze and excitation. Default False.")
    parser.add_argument("--use_initial_conv", action="store_true",
                        help="Whether to use the initial 7x7 kernel convolution. Default False.")
    parser.add_argument("--use_residual_atrous_conv", action="store_true",
                        help="Whether to use residual atrous convolutions. Default False.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[2, 3, 4, 6, 6, 7, 7],
                        help="Number of hidden blocks for ResNets. Ignored if not resnet.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--bottleneck_expansion", type=int, default=1,
                        help="The expansion factor of the bottleneck. Default 1.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true",
                        help="Whether to use attention in the U-Net. Default False. Cannot be used with unet_plus.")
    parser.add_argument("--use_separated_background", action="store_true",
                        help="Whether to use a separated outconv for background sigmoid. Default False. Must be used with atrous conv.")
    parser.add_argument("--image_stain_norm", action="store_true",
                        help="Whether to stain normalize the images. Default False.")
    parser.add_argument("--disable_tta", action="store_true", help="Whether to disable TTA. Default False.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to use. Should be a .pt file")
    parser.add_argument("--out_hdf5_file", type=str, required=True, help="Path to the output HDF5 file.")
    parser.add_argument("--inference_on_train", action="store_true", help="Whether to run inference on the training set. Used for debug. Default False.")
    args = parser.parse_args()

    blocks = args.hidden_blocks
    assert (not args.use_separated_background) or args.use_atrous_conv, "Must use atrous conv if using separated background"
    use_separated_background = args.use_separated_background
    use_residual_atrous_conv = args.use_residual_atrous_conv
    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                    hidden_channels=args.hidden_channels,
                                                    use_batch_norm=args.use_batch_norm,
                                                    use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                    in_channels=4, use_atrous_conv=args.use_atrous_conv,
                                                    atrous_outconv_split=use_separated_background,
                                                    atrous_outconv_residual=use_residual_atrous_conv,
                                                    deep_supervision=True,
                                                    squeeze_excitation=args.use_squeeze_excitation,
                                                    bottleneck_expansion=args.bottleneck_expansion,
                                                    res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(device=config.device)
    else:
        model = model_unet_base.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                               hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                               deep_supervision=True,
                                               in_channels=4, use_atrous_conv=args.use_atrous_conv,
                                               atrous_outconv_split=use_separated_background,
                                               atrous_outconv_residual=use_residual_atrous_conv,
                                               squeeze_excitation=args.use_squeeze_excitation,
                                               bottleneck_expansion=args.bottleneck_expansion,
                                               res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(device=config.device)

    model_checkpoint_path = args.model_path
    model.load_state_dict(torch.load(model_checkpoint_path))

    computed = 0
    last_compute_print = 0
    ctime = time.time()

    image_radius = 384

    inference_on_train = args.inference_on_train
    if inference_on_train:
        inference_data = [tile_id[:-4] for tile_id in os.listdir(os.path.join(config.input_data_path, "train"))]
        inference_data = np.random.choice(inference_data, size=int(len(inference_data) * 0.01), replace=False)
    else:
        inference_data = [tile_id[:-4] for tile_id in os.listdir(os.path.join(config.input_data_path, "test"))]

    hdf5_file = h5py.File(args.out_hdf5_file, mode="w")

    print("Computing the logits now...")
    print("Using stain normalization: {}".format(args.image_stain_norm))
    print("Using separated background: {}".format(use_separated_background))
    print("Disable TTA: {}".format(args.disable_tta))
    print("DOING INFERENCE ON TRAINING SET: {}".format(inference_on_train))

    disable_tta = args.disable_tta
    with tqdm.tqdm(total=len(inference_data)) as pbar:
        while computed < len(inference_data):
            tile_id = inference_data[computed]
            # Compute logits
            with torch.no_grad():
                img_helper = inference_reconstructed_base.Composite1024To512ImageInference()
                img_helper.load_image(tile_id, stain_normalize=args.image_stain_norm)
                for location in inference_reconstructed_base.Composite1024To512ImageInference.LOCATIONS:
                    if disable_tta and (location != inference_reconstructed_base.Composite1024To512ImageInference.CENTER):
                        continue
                    img = img_helper.get_combined_image(location)
                    if disable_tta:
                        logits_tensor = get_result_logits(model, img.unsqueeze(0), test_time_augmentation=False).permute(2, 3, 1, 0)
                    else:
                        logits_tensor = get_result_logits(model, img.unsqueeze(0), test_time_augmentation=True)
                    img_helper.store_prediction_logits(location, logits_tensor)
                result = img_helper.obtain_predictions(reduction_logit_average=False, experts_only=False, separated_logits=use_separated_background, center_only=False)
                hdf5_file.create_dataset(tile_id, data=result.cpu().numpy(), compression="gzip", compression_opts=4)
                del img_helper, result

            gc.collect()
            pbar.update(1)
            computed += 1

            if computed - last_compute_print >= 100:
                print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
                last_compute_print = computed
                ctime = time.time()

    print("Inference Complete")
    hdf5_file.close()
