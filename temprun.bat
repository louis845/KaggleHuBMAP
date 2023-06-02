REM python model_simple_unet.py --model_name unet_batchnorm --epochs 240 --learning_rate 3e-7 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 2 --gradient_accumulation_steps 4 --epochs_per_save 10 --use_batch_norm
python model_simple_unet.py --model_name unet_rotation_batchnorm_grayscale --epochs 240 --rotation_augmentation --learning_rate 1e-5 --dataset grayscale_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 2 --gradient_accumulation_steps 4 --epochs_per_save 10 --use_batch_norm --use_res_conv --hidden_channels 24 --in_channels 1
REM python inference_simple_unet.py --model_name unet_rotation_batchnorm_continue --original_data_name hue_shifted_images --transformed_data_name unet_segmentation_mask_2 --batch_size 8 --use_batch_norm --use_res_conv --hidden_channels 24
REM python inference_simple_unet.py --model_name unet_height_6_continue --original_data_name hue_shifted_images --transformed_data_name unet_segmentation_mask_height6_2 --batch_size 2 --use_batch_norm --pyramid_height 6

REM python model_autoencoder.py --model_name autoencoder --epochs 20 --rotation_augmentation --learning_rate 3e-7 --dataset hue_shifted_images --train_subdata dataset_half_stratify_wsi --val_subdata dataset_half_stratify_wsi_2 --batch_size 3 --gradient_accumulation_steps 3 --use_batch_norm

REM python model_pixelwise_encoder_decoder.py --epochs 750 --batch_size 8 --learning_rate 3e-3 --model_name pixelwise_encoder_decoder --train_subdata dataset_50_restrict_wsi_1_2_stratify_wsi --val_subdata dataset_50_restrict_wsi_1_2_stratify_wsi_2

REM python model_simple_unet.py --model_name unet_rotation_batchnorm_continue --epochs 400 --rotation_augmentation --learning_rate 1e-6 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 2 --gradient_accumulation_steps 4 --epochs_per_save 10 --use_batch_norm --use_res_conv --hidden_channels 24 --prev_model_ckpt unet_rotation_batchnorm

REM python model_simple_unet.py --model_name unet_plus_plus --epochs 4 --rotation_augmentation --learning_rate 1e-5 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 1 --gradient_accumulation_steps 4 --epochs_per_save 10 --use_batch_norm --hidden_channels 3 --unet_plus standard

REM python model_simple_unet.py --model_name unet_plus_plus_small --epochs 50 --rotation_augmentation --learning_rate 1e-4 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 1 --gradient_accumulation_steps 4 --epochs_per_save 25 --use_batch_norm --pyramid_height 4 --hidden_channels 48 --unet_plus deep_supervision
REM python model_simple_unet.py --model_name unet_plus_plus_small_continue --epochs 100 --rotation_augmentation --learning_rate 1e-5 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 1 --gradient_accumulation_steps 4 --epochs_per_save 25 --use_batch_norm --pyramid_height 4 --hidden_channels 48 --unet_plus deep_supervision --prev_model_ckpt unet_plus_plus_small
REM python model_simple_unet.py --model_name unet_plus_plus_small_continue2 --epochs 250 --rotation_augmentation --learning_rate 1e-6 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 1 --gradient_accumulation_steps 4 --epochs_per_save 25 --use_batch_norm --pyramid_height 4 --hidden_channels 48 --unet_plus deep_supervision --prev_model_ckpt unet_plus_plus_small_continue