python model_simple_unet.py --model_name simple_unet --epochs 20 --learning_rate 1e-6 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 2 --gradient_accumulation_steps 4

python model_simple_unet.py --model_name simple_unet_rotation --epochs 20 --rotation_augmentation --learning_rate 1e-6 --dataset hue_shifted_images --train_subdata dataset1_split1 --val_subdata dataset1_split2 --batch_size 2 --gradient_accumulation_steps 4

REM python model_pixelwise_encoder_decoder.py --epochs 750 --batch_size 8 --learning_rate 3e-3 --model_name pixelwise_encoder_decoder --train_subdata dataset_50_restrict_wsi_1_2_stratify_wsi --val_subdata dataset_50_restrict_wsi_1_2_stratify_wsi_2