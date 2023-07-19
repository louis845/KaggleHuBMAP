python inference_reconstructed_unet.py --batch_size 1 --model_name unet_reconstructed_progressive_multiclass_height6_atrous_separated --transformed_data_name atrous_separated_rough_notta_conf --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type confidence --use_batch_norm --use_res_conv --use_atrous_conv --use_squeeze_excitation --use_initial_conv --hidden_blocks 1 3 8 23 10 4 4 --hidden_channels 20 --bottleneck_expansion 4 --pyramid_height 6 --unet_attention --use_separated_background

REM python inference_reconstructed_instances_from_logits.py --separated_logits --subdata dataset1 --original_data_name atrous_separated_rough_logits --transformed_data_name atrous_separated_rough_instances

REM python inference_reconstructed_unet_from_logits.py --separated_logits --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type argmax --original_data_name atrous_separated_rough_logits --transformed_data_name atrous_separated_rough_argmax
REM python inference_reconstructed_unet_from_logits.py --separated_logits --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type confidence --original_data_name atrous_separated_rough_logits --transformed_data_name atrous_separated_rough_conf
REM python inference_reconstructed_unet_from_logits.py --separated_logits --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type noconfidence --original_data_name atrous_separated_rough_logits --transformed_data_name atrous_separated_rough_noconf
REM python inference_reconstructed_unet_from_logits.py --separated_logits --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --prediction_type levels --original_data_name atrous_separated_rough_logits --transformed_data_name atrous_separated_rough_levels

REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type argmax --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_argmax
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type confidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_conf
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type noconfidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_noconf
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --prediction_type levels --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_lavg_levels

REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type argmax --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_argmax
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type confidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_conf
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type noconfidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_noconf
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --experts_only --prediction_type levels --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_levels

REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type argmax --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_argmax
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type confidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_conf
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type noconfidence --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_noconf
REM python inference_reconstructed_unet_from_logits.py --subdata dataset1 --train_subdata dataset1_regional_split1 --val_subdata dataset1_regional_split2 --reduction_logit_average --experts_only --prediction_type levels --original_data_name atrous_cbs_more_logits --transformed_data_name atrous_cbs_more_expert_lavg_levels