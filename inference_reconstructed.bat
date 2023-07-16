REM python inference_reconstructed_logits.py --model_name unet_reconstructed_progressive_multiclass_height6_atrous_cbs_more5 --transformed_data_name atrous_cbs_more_logits --subdata dataset1 --use_batch_norm --use_res_conv --use_atrous_conv --use_squeeze_excitation --use_initial_conv --hidden_blocks 1 3 6 8 8 14 14 --hidden_channels 20 --bottleneck_expansion 4 --pyramid_height 6 --unet_attention

python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type argmax --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_argmax
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type confidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_conf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type noconfidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_noconf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type levels --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_levels

python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type argmax --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_argmax
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type confidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_conf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type noconfidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_noconf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type levels --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_levels

python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type argmax --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_argmax
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type confidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_conf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type noconfidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_noconf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type levels --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_levels

python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type argmax --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_argmax
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type confidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_conf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type noconfidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_noconf
python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type levels --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_levels