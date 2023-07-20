"""Obtain the raw logits from the model. The model here is trained with reconstructed_model_progressive_supervised_unet.py"""
import collections
import gc
import os
import time
import argparse

import config

import tqdm

import numpy as np
import torch.nn
import cv2

import model_data_manager
import model_unet_base
import model_unet_attention
import inference_reconstructed_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnosis of a multiclass reconstructed U-Net model")
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
    parser.add_argument("--samples", type=int, default=10, help="Number of sample tile_ids for diagnosis. Default 10.")

    model_data_manager.transform_add_argparse_arguments(parser)

    args = parser.parse_args()

    input_data_loader, output_data_writer, model_path, subdata_entries, train_subdata_entries, val_subdata_entries = model_data_manager.transform_get_argparse_arguments(args)

    blocks = args.hidden_blocks
    assert (not args.use_separated_background) or args.use_atrous_conv, "Must use atrous conv if using separated background"
    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                    hidden_channels=args.hidden_channels,
                                                    use_batch_norm=args.use_batch_norm,
                                                    use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                    in_channels=4, use_atrous_conv=args.use_atrous_conv, atrous_outconv_split=args.use_separated_background,
                                                    deep_supervision=True,
                                                    squeeze_excitation=args.use_squeeze_excitation,
                                                    bottleneck_expansion=args.bottleneck_expansion,
                                                    res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(
            device=config.device)
    else:
        model = model_unet_base.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                               hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                               in_channels=4, use_atrous_conv=args.use_atrous_conv, atrous_outconv_split=args.use_separated_background,
                                               deep_supervision=True,
                                               squeeze_excitation=args.use_squeeze_excitation,
                                               bottleneck_expansion=args.bottleneck_expansion,
                                               res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(
            device=config.device)

    model_checkpoint_path = os.path.join(model_path, "model.pt")

    model.load_state_dict(torch.load(model_checkpoint_path))

    computed = 0
    last_compute_print = 0
    ctime = time.time()

    image_radius = 384
    # randomly draw args.samples number of tiles
    subdata_entries = np.random.choice(subdata_entries, size=args.samples, replace=False)

    all_encoders_diagnosis_channel = collections.defaultdict(lambda: np.array([], dtype=np.float32))
    all_encoders_diagnosis_pixel = collections.defaultdict(lambda: np.array([], dtype=np.float32))
    all_classifiers_diagnosis_channel = collections.defaultdict(lambda: np.array([], dtype=np.float32))
    all_classifiers_diagnosis_pixel = collections.defaultdict(lambda: np.array([], dtype=np.float32))

    print("Computing the logits now...")
    with tqdm.tqdm(total=len(subdata_entries)) as pbar:
        while computed < len(subdata_entries):
            tile_id = subdata_entries[computed]
            # Compute diagnosis summary
            encoder_diagnosis = []
            classifier_diagnosis = []
            with torch.no_grad():
                single_image_batch = inference_reconstructed_base.load_combined(tile_id).unsqueeze(0)
                diagnosis_result = model(single_image_batch, diagnosis=True)
                x_list = diagnosis_result["encoder"]
                if args.unet_attention:
                    result, attention_layers, deep_outputs, diagnosis_outputs = diagnosis_result["classifier"]
                else:
                    result, deep_outputs, diagnosis_outputs = diagnosis_result["classifier"]

                for x in x_list:
                    # x is a (1, C, H, W) tensor.
                    # standard deviation across pixels in each channel
                    across_pixel_std = torch.std(x, dim=(2, 3)).squeeze(0)
                    # standard deviation across channels in each pixel
                    across_channel_std = torch.std(x, dim=(1,)).squeeze(0)
                    across_channel_std = across_channel_std.reshape(-1)
                    encoder_diagnosis.append({"channel_std": across_channel_std.cpu().numpy(), "pixel_std": across_pixel_std.cpu().numpy()})

                for x in diagnosis_outputs:
                    # x is a (1, C, H, W) tensor.
                    # standard deviation across pixels in each channel
                    across_pixel_std = torch.std(x, dim=(2, 3)).squeeze(0)
                    # standard deviation across channels in each pixel
                    across_channel_std = torch.std(x, dim=(1,)).squeeze(0)
                    across_channel_std = across_channel_std.reshape(-1)
                    classifier_diagnosis.append({"channel_std": across_channel_std.cpu().numpy(), "pixel_std": across_pixel_std.cpu().numpy()})

                if args.unet_attention:
                    for k in range(len(attention_layers)):
                        attention_map = attention_layers[k] # (1, 1, H, W) tensor.
                        # convert to numpy and save as image "{}_attention{}.png".format(tile_id, k) insider output_data_writer.data_folder
                        attention_map = attention_map.squeeze(0).squeeze(0).cpu().numpy()
                        attention_map = (attention_map * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(output_data_writer.data_folder, "{}_attention{}.png".format(tile_id, k)), attention_map)

                # save the diagnosis results
                for k in range(len(encoder_diagnosis)):
                    all_encoders_diagnosis_channel[k] = np.concatenate([all_encoders_diagnosis_channel[k], encoder_diagnosis[k]["channel_std"]])
                    all_encoders_diagnosis_pixel[k] = np.concatenate([all_encoders_diagnosis_pixel[k], encoder_diagnosis[k]["pixel_std"]])
                for k in range(len(classifier_diagnosis)):
                    all_classifiers_diagnosis_channel[k] = np.concatenate([all_classifiers_diagnosis_channel[k], classifier_diagnosis[k]["channel_std"]])
                    all_classifiers_diagnosis_pixel[k] = np.concatenate([all_classifiers_diagnosis_pixel[k], classifier_diagnosis[k]["pixel_std"]])

            gc.collect()
            pbar.update(1)
            computed += 1

            if computed - last_compute_print >= 100:
                print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
                last_compute_print = computed
                ctime = time.time()

    diagnosis_group = output_data_writer.data_store.create_group("diagnosis")  # h5py.Group
    for k in range(len(all_encoders_diagnosis_channel)):
        diagnosis_group.create_dataset("encoder_channel_{}".format(k), data=all_encoders_diagnosis_channel[k],
                                       compression="gzip")
        diagnosis_group.create_dataset("encoder_pixel_{}".format(k), data=all_encoders_diagnosis_pixel[k],
                                       compression="gzip")
    for k in range(len(all_classifiers_diagnosis_channel)):
        diagnosis_group.create_dataset("classifier_channel_{}".format(k), data=all_classifiers_diagnosis_channel[k],
                                       compression="gzip")
        diagnosis_group.create_dataset("classifier_pixel_{}".format(k), data=all_classifiers_diagnosis_pixel[k],
                                       compression="gzip")

    input_data_loader.close()
    output_data_writer.close()
    print("Inference Complete")